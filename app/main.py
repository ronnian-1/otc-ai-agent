"""Simple CLI entry point for the OTC Agent prototype.

Run with ``python -m app.main`` from the project root. The CLI delegates Agent,
Tool, Memory, and Safety behavior to their existing modules and only handles
interactive input/output plus logging.
"""

from __future__ import annotations

from app.agent.agent import build_agent, run_agent_turn
from app.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    """Start a minimal interactive CLI for local demos."""
    try:
        runtime = build_agent()
    except Exception:
        # Terminal logs must not include provider exceptions or tracebacks.
        logger.error("Agent initialization failed")
        print("Agent 初始化失败。")
        print("请确认 .env 中已配置 OPENAI_API_KEY，或在测试中使用 Mock LLM。")
        return

    logger.info("OTC Agent CLI started")
    print("天猫国际 OTC / 保健品 AI 导购 Agent 已启动。输入 exit 或 quit 退出。")

    while True:
        try:
            user_input = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            logger.info("OTC Agent CLI stopped by user")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("已退出。")
            logger.info("OTC Agent CLI exited")
            break

        if not user_input:
            continue

        try:
            logger.info("Agent turn started")
            answer = run_agent_turn(runtime, user_input)
            logger.info("Agent turn completed")
            print(f"Agent：{answer}")
        except Exception:
            logger.error("Agent turn failed")
            print("Agent 执行失败，请稍后重试。")


if __name__ == "__main__":
    main()

