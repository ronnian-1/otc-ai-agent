"""CLI control-flow and error-disclosure tests with synthetic dependencies."""

import logging
from unittest.mock import Mock, call

import pytest

from app import main as cli


SECRET = "api_key=DEMO_SECRET_NOT_REAL"
RAW_ERROR = f"Synthetic transport failure: {SECRET}; private request context"


@pytest.fixture
def cli_logs(caplog):
    """Capture CLI records, including any attached exception information."""
    caplog.set_level(logging.INFO, logger=cli.logger.name)
    return caplog


def _run_cli():
    """Bind a real stderr handler during the test call's capture lifetime."""
    handler = logging.StreamHandler()
    cli.logger.addHandler(handler)
    try:
        return cli.main()
    finally:
        cli.logger.removeHandler(handler)
        handler.close()


def _assert_no_error_details(captured, caplog):
    for output in (captured.out, captured.err, caplog.text):
        assert SECRET not in output
        assert RAW_ERROR not in output
        assert "private request context" not in output
        assert "Traceback" not in output
    for record in caplog.records:
        assert record.exc_info is None
        assert record.exc_text is None
        assert record.stack_info is None


def test_cli_initialization_failure_is_sanitized_and_returns(
    monkeypatch, capfd, cli_logs
):
    build = Mock(side_effect=RuntimeError(RAW_ERROR))
    read_input = Mock(side_effect=AssertionError("must not start input loop"))
    run_turn = Mock()
    monkeypatch.setattr(cli, "build_agent", build)
    monkeypatch.setattr(cli, "run_agent_turn", run_turn)
    monkeypatch.setattr("builtins.input", read_input)

    assert _run_cli() is None

    captured = capfd.readouterr()
    _assert_no_error_details(captured, cli_logs)
    assert "Agent 初始化失败" in captured.out
    assert "Agent initialization failed" in captured.err
    assert "已启动" not in captured.out
    assert "OTC Agent CLI started" not in cli_logs.text
    assert [record.getMessage() for record in cli_logs.records] == [
        "Agent initialization failed"
    ]
    assert cli_logs.records[0].levelno == logging.ERROR
    build.assert_called_once_with()
    read_input.assert_not_called()
    run_turn.assert_not_called()


def test_cli_turn_failure_is_sanitized_and_next_turn_can_run(
    monkeypatch, capfd, cli_logs
):
    runtime = object()
    build = Mock(return_value=runtime)
    run_turn = Mock(side_effect=[RuntimeError(RAW_ERROR), "合成正常回复"])
    read_input = Mock(side_effect=["S001价格", "S002规格", "exit"])
    monkeypatch.setattr(cli, "build_agent", build)
    monkeypatch.setattr(cli, "run_agent_turn", run_turn)
    monkeypatch.setattr("builtins.input", read_input)

    assert _run_cli() is None

    captured = capfd.readouterr()
    _assert_no_error_details(captured, cli_logs)
    assert "Agent 执行失败" in captured.out
    assert "Agent turn failed" in captured.err
    assert "Agent：合成正常回复" in captured.out
    assert "已退出。" in captured.out
    assert captured.out.index("Agent 执行失败") < captured.out.index("合成正常回复")
    assert [record.getMessage() for record in cli_logs.records] == [
        "OTC Agent CLI started",
        "Agent turn started",
        "Agent turn failed",
        "Agent turn started",
        "Agent turn completed",
        "OTC Agent CLI exited",
    ]
    assert cli_logs.records[2].levelno == logging.ERROR
    build.assert_called_once_with()
    assert run_turn.call_args_list == [
        call(runtime, "S001价格"), call(runtime, "S002规格")
    ]


@pytest.mark.parametrize("exit_command", ["exit", "quit", " EXIT ", "QuIt"])
def test_cli_normal_turn_empty_input_and_exit(
    monkeypatch, capfd, cli_logs, exit_command
):
    runtime = object()
    build = Mock(return_value=runtime)
    run_turn = Mock(return_value="合成商品回复")
    read_input = Mock(side_effect=["", "   ", " S001价格 ", exit_command])
    monkeypatch.setattr(cli, "build_agent", build)
    monkeypatch.setattr(cli, "run_agent_turn", run_turn)
    monkeypatch.setattr("builtins.input", read_input)

    assert _run_cli() is None

    captured = capfd.readouterr()
    assert "已启动" in captured.out
    assert "Agent：合成商品回复" in captured.out
    assert "已退出。" in captured.out
    assert "失败" not in captured.out
    assert "OTC Agent CLI exited" in cli_logs.text
    build.assert_called_once_with()
    run_turn.assert_called_once_with(runtime, "S001价格")
    assert read_input.call_count == 4


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
def test_cli_input_interruption_exits_cleanly(
    monkeypatch, capfd, cli_logs, interruption
):
    monkeypatch.setattr(cli, "build_agent", Mock(return_value=object()))
    run_turn = Mock()
    monkeypatch.setattr(cli, "run_agent_turn", run_turn)
    monkeypatch.setattr("builtins.input", Mock(side_effect=interruption))

    assert _run_cli() is None

    captured = capfd.readouterr()
    assert "已退出。" in captured.out
    assert "失败" not in captured.out
    assert "Traceback" not in captured.err
    assert "OTC Agent CLI stopped by user" in cli_logs.text
    run_turn.assert_not_called()


def test_cli_keyboard_interrupt_during_turn_still_propagates(monkeypatch, cli_logs):
    """Do not silently turn a processing interrupt into a recoverable failure."""
    monkeypatch.setattr(cli, "build_agent", Mock(return_value=object()))
    run_turn = Mock(side_effect=KeyboardInterrupt)
    read_input = Mock(return_value="S001价格")
    monkeypatch.setattr(cli, "run_agent_turn", run_turn)
    monkeypatch.setattr("builtins.input", read_input)

    with pytest.raises(KeyboardInterrupt):
        _run_cli()

    read_input.assert_called_once()
    run_turn.assert_called_once()
    assert "Agent turn completed" not in cli_logs.text
    assert "Agent turn failed" not in cli_logs.text
