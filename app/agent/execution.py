"""Per-invocation execution limits at the model/tool dispatch boundaries.

These are local engineering limits, not provider retry or wall-clock limits.
ToolNode can dispatch a batch concurrently; reservations and failures are atomic.
"""

from __future__ import annotations

import json
import math
from collections import deque
from concurrent.futures import CancelledError, Future, TimeoutError as FutureTimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from httpx import TimeoutException
from openai import APITimeoutError
from typing_extensions import NotRequired

from app.agent.reply import (
    EXECUTION_STOP_REPLY, EXECUTION_TIMEOUT_REPLY, EXECUTION_BUSY_REPLY,
    EXECUTION_POLICY_STOP_REPLY,
)


MAX_MODEL_CALLS = 8
MAX_TOOL_CALLS = 12
MAX_INVALID_CALLS = 3
MAX_TOOL_ERRORS = 2


@dataclass(frozen=True)
class TimeoutPolicy:
    """Central policy; elapsed time uses a monotonic clock, never fact TTL time.

    The turn starts at run_agent_turn entry (at before_agent for direct invoke).
    Deterministic preparation/reviews count toward it but are cooperative: final
    safety review and history commit must finish even after the deadline.
    """
    tool_seconds: float = 5.0
    model_seconds: float = 30.0
    turn_seconds: float = 60.0
    tool_workers: int = 4

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in
               (self.tool_seconds, self.model_seconds, self.turn_seconds)):
            raise ValueError("Timeouts must be positive finite seconds")
        if not isinstance(self.tool_workers, int) or not 1 <= self.tool_workers <= MAX_TOOL_CALLS:
            raise ValueError("Invalid read-only worker capacity")


DEFAULT_TIMEOUTS = TimeoutPolicy()


class ExecutionStopped(Exception):
    """Control signal, containing no provider/tool exception text."""


class _Workers:
    """Bounded daemon workers: never join an uncancellable synchronous call.

    Only raw read-only tool/model callbacks run here, not facts/history commits.
    Threads exit when the queue drains. Pending Futures can be cancelled;
    running functions cannot. Capacity persists across turns of one runtime.
    """
    def __init__(self, capacity, pending):
        self.capacity, self.pending = capacity, pending
        self.lock = Lock()
        self.active = 0
        self.queue = deque()
        self.running = set()
        self.idle = Event()
        self.idle.set()

    def submit(self, callback):
        context, future = copy_context(), Future()
        with self.lock:
            self.queue = deque(item for item in self.queue if not item[0].cancelled())
            if len(self.running) + len(self.queue) >= self.capacity + self.pending:
                return None
            self.queue.append((future, context, callback))
            self.idle.clear()
            if self.active < self.capacity:
                self.active += 1
                try:
                    Thread(target=self._work, name="otc-read-only-call", daemon=True).start()
                except BaseException:
                    # Thread creation/start can fail under resource pressure.
                    # Roll back while holding the lock: no worker has taken this
                    # item, and a nonexistent thread must not consume capacity.
                    self.active -= 1
                    self.queue.remove((future, context, callback))
                    future.cancel()
                    if not self.active:
                        self.idle.set()
                    raise
        return future

    def _work(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.active -= 1
                    if not self.active:
                        self.idle.set()
                    return
                future, context, callback = self.queue.popleft()
                self.running.add(future)
            if future.set_running_or_notify_cancel():
                try:
                    value = context.run(callback)
                except BaseException as exc:
                    with self.lock:
                        self.running.remove(future)
                    future.set_exception(exc)
                else:
                    with self.lock:
                        self.running.remove(future)
                    future.set_result(value)
            else:
                with self.lock:
                    self.running.remove(future)


class _Budget:
    def __init__(self, policy=None, workers=None):
        self.lock = Lock()
        self.policy = policy or DEFAULT_TIMEOUTS
        self.workers = workers or {"tool": _Workers(self.policy.tool_workers, MAX_TOOL_CALLS),
                                   "model": _Workers(1, 0)}
        self.started = monotonic()
        self.deadline = self.started + self.policy.turn_seconds
        self.waits = []
        self.pending = []
        self.started_calls = {"model": 0, "tool": 0}
        self.interrupted = False
        self.models = self.tools = self.invalid = self.errors = 0
        self.reason = None
        self.events = []
        self.scopes = []
        self._pending_policy_denials = {}
        self.policy_errors = 0

    def _expire(self):
        if monotonic() >= self.deadline:
            self.reason = self.reason or "turn_timeout"
            self.interrupted = True

    def stop(self, reason):
        with self.lock:
            self._expire()
            self.reason = self.reason or reason  # First observed stop wins.
            self.interrupted |= "timeout" in reason or reason.endswith(("_busy", "_unavailable"))
            for future in self.pending:
                future.cancel()

    def check(self, *, admitted=False):
        with self.lock:
            self._expire()
            # Preserve phase-5 semantics: already admitted calls may finish a
            # count-budget batch. Timeouts always abandon queued/pending reads.
            if self.reason and not (admitted and not self.interrupted and self.reason in (
                    "tool_budget", "invalid_call_budget", "tool_error_budget")):
                raise ExecutionStopped()

    def exit_reply(self):
        if self.reason in ("model_busy", "tool_busy"):
            return EXECUTION_BUSY_REPLY
        if self.reason == "tool_error_budget" and 0 < self.errors == self.policy_errors:
            return EXECUTION_POLICY_STOP_REPLY
        return EXECUTION_TIMEOUT_REPLY if self.reason and "timeout" in self.reason else EXECUTION_STOP_REPLY

    def note_policy_denial(self, call_id):
        # Only the pre-execution policy gate registers these; tool JSON cannot.
        with self.lock:
            self._pending_policy_denials[call_id] = self._pending_policy_denials.get(call_id, 0) + 1

    def touch_scope(self, name, args):
        with self.lock:
            self.scopes.append((name, args))

    def commit_fact(self, callback):
        # No late worker can write: callbacks that mutate facts run only here.
        with self.lock:
            self._expire()
            if self.reason:
                raise ExecutionStopped()
            return callback()

    def run(self, kind, callback, call_id):
        """Wait at most min(call timeout, remaining turn time); discard late work."""
        self.check(admitted=kind == "tool")
        call_deadline = monotonic() + getattr(self.policy, kind + "_seconds")
        def checked_callback():
            self.check(admitted=kind == "tool")
            with self.lock:
                self.started_calls[kind] += 1
            return callback()
        try:
            future = self.workers[kind].submit(checked_callback)
        except Exception:
            # Dispatch failed before a callback started. Do not expose OS/provider
            # details, retry a broken dispatcher, or claim an older call is busy.
            self.stop(kind + "_unavailable")
            raise ExecutionStopped() from None
        if future is None:
            self.stop(kind + "_busy")
            raise ExecutionStopped()
        with self.lock:
            self.pending.append(future)
        try:
            while True:
                self.check(admitted=kind == "tool")
                remaining = min(self.deadline, call_deadline) - monotonic()
                if remaining <= 0:
                    self.stop("turn_timeout" if self.deadline <= call_deadline else kind + "_timeout")
                    raise ExecutionStopped()
                try:
                    value = future.result(timeout=min(0.01, remaining))
                except CancelledError:
                    self.check()  # Cancellation by a peer's timeout is a stop.
                    self.stop(kind + "_timeout")
                    raise ExecutionStopped() from None
                except (FutureTimeoutError, TimeoutError, TimeoutException, APITimeoutError):
                    if not future.done():
                        continue
                    if future.cancelled():
                        self.check()
                        self.stop(kind + "_timeout")
                        raise ExecutionStopped() from None
                    error = future.exception()
                    if isinstance(error, (FutureTimeoutError, TimeoutError, TimeoutException, APITimeoutError)):
                        self.stop(kind + "_timeout")
                        raise ExecutionStopped() from None
                    if error is not None:
                        raise error
                    # A polling wait can expire just before a successful call
                    # completes. Completion alone does not prove call timeout.
                    value = future.result()
                self.check(admitted=kind == "tool")
                if monotonic() >= call_deadline:
                    self.stop(kind + "_timeout")
                    raise ExecutionStopped()
                return value
        except ExecutionStopped:
            cancelled = future.cancel()
            action = ("cancelled_before_start" if cancelled else
                      "finished_discarded" if future.done() else "running_not_cancelled")
            with self.lock:
                self.waits.append({"kind": kind, "call_id": call_id, "action": action})
            raise
        finally:
            with self.lock:
                self.pending.remove(future)

    def enter_model(self):
        with self.lock:
            self._expire()
            if not self.reason and self.models >= MAX_MODEL_CALLS:
                self.reason = "model_budget"
            if self.reason:
                return False
            self.models += 1
            return True

    def reserve_tool(self, call):
        with self.lock:
            self._expire()
            if not self.reason and self.tools >= MAX_TOOL_CALLS:
                self.reason = "tool_budget"
            admitted = not self.reason
            if admitted:
                self.tools += 1
            self.events.append({"call_id": call["id"], "tool": call["name"],
                                "outcome": "admitted" if admitted else "blocked_budget"})
            return admitted

    def finish_tool(self, call, result):
        try:
            payload = json.loads(result.content)
        except (ValueError, TypeError, AttributeError):
            payload = {}
        check = payload.get("_task_check", {}) if isinstance(payload, dict) else {}
        evidence = payload.get("_tool_evidence", {}) if isinstance(payload, dict) else {}
        invalid = (check.get("execution_args", True) is None
                   or (isinstance(payload, dict) and payload.get("error") == "invalid_tool_schema")
                   or call["name"] not in ("search_product", "compare_product"))
        stopped = isinstance(payload, dict) and payload.get("error") == "execution_stopped"
        error = not invalid and not stopped and (not isinstance(result, ToolMessage) or result.status == "error")
        outcome = ("stopped" if stopped else "invalid" if invalid else "tool_error" if error
                   else evidence.get("status", "success"))
        with self.lock:
            policy_denials = self._pending_policy_denials.get(call["id"], 0)
            if policy_denials:
                if policy_denials == 1:
                    del self._pending_policy_denials[call["id"]]
                else:
                    self._pending_policy_denials[call["id"]] -= 1
                self.policy_errors += int(error)
            else:
                self.scopes.append((call["name"], evidence.get("query_args", call["args"])))
            self.invalid += int(invalid)
            self.errors += int(error)
            for event in self.events:
                if event["call_id"] == call["id"] and event["outcome"] == "admitted":
                    event["outcome"] = outcome
                    break
            if not self.reason:
                if self.invalid >= MAX_INVALID_CALLS:
                    self.reason = "invalid_call_budget"
                elif self.errors >= MAX_TOOL_ERRORS:
                    self.reason = "tool_error_budget"

    def summary(self):
        with self.lock:
            self._expire()
            return {"model_calls": self.models, "tool_calls": self.tools,
                    "invalid_calls": self.invalid, "tool_errors": self.errors,
                    "stopped": bool(self.reason), "reason": self.reason,
                    "events": [dict(event) for event in self.events],
                    "waits": [dict(wait) for wait in self.waits],
                    "calls_started": dict(self.started_calls),
                    "elapsed_seconds": monotonic() - self.started,
                    "exit_reply": self.exit_reply() if self.reason else None}


class _ExecutionState(AgentState):
    execution_budget: NotRequired[Any]
    execution: NotRequired[dict[str, Any]]


class ExecutionMiddleware(AgentMiddleware):
    """Stop before another model/dispatch, then quarantine aborted evidence.

    Count limits permit admitted calls to finish; timeouts abandon their waits.
    after_agent joins the bounded waiters, then quarantines touched facts. Late
    raw callbacks cannot write facts. Invocation budgets are separate from the
    shared bounded workers and public-turn lock.
    """

    state_schema = _ExecutionState

    def __init__(self, memory, policy=None):
        self.memory = memory
        self.policy = policy or DEFAULT_TIMEOUTS
        self.workers = {"tool": _Workers(self.policy.tool_workers, MAX_TOOL_CALLS), "model": _Workers(1, 0)}
        self.turn_lock = Lock()

    def new_budget(self):
        return _Budget(self.policy, self.workers)

    def before_agent(self, state, runtime):
        return {"execution_budget": state.get("execution_budget") or self.new_budget(), "execution": {}}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if not state["execution_budget"].enter_model():
            return {"messages": [AIMessage(content=state["execution_budget"].exit_reply())], "jump_to": "end"}
        return None

    def wrap_model_call(self, request, handler):
        budget = request.state["execution_budget"]
        try:
            return budget.run("model", lambda: handler(request), "model_" + str(budget.models))
        except ExecutionStopped:
            return AIMessage(content=budget.exit_reply())

    async def abefore_agent(self, state, runtime):
        # The fact middleware and public turn API are synchronous. Fail closed
        # before dispatch rather than silently bypassing their checks in ainvoke.
        raise RuntimeError("Async Agent execution is unsupported; use run_agent_turn")

    def wrap_tool_call(self, request, handler):
        budget = request.state["execution_budget"]
        call = request.tool_call
        if not budget.reserve_tool(call):
            return ToolMessage(content=json.dumps({"error": "execution_budget_exhausted",
                                                  "message": "本轮已停止，不得宣称查询成功。"}),
                               name=call["name"], tool_call_id=call["id"], status="error")
        try:
            result = handler(request)
        except ExecutionStopped:
            result = ToolMessage(content=json.dumps({"error": "execution_stopped"}),
                                 name=call["name"], tool_call_id=call["id"], status="error")
        except ValidationError:
            result = ToolMessage(content=json.dumps({"error": "invalid_tool_schema",
                                                    "message": "参数未通过工具 Schema，请按工具定义重新规划。"}),
                                 name=call["name"], tool_call_id=call["id"], status="error")
        except Exception:
            # Provider/tool exception text is not safe model or user evidence.
            result = ToolMessage(content=json.dumps({"error": "tool_execution_failed",
                                                    "message": "本调用失败，可核对参数后重试；不得推断商品事实。"}),
                                 name=call["name"], tool_call_id=call["id"], status="error")
        budget.finish_tool(call, result)
        return result

    def after_agent(self, state, runtime):
        budget = state["execution_budget"]
        summary = budget.summary()
        if summary["stopped"]:
            self.memory.facts.discard_queries(budget.scopes)
        return {"execution_budget": None, "execution": summary}
