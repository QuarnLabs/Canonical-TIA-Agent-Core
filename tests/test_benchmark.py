from canonical_tia_core.benchmarks.interface_battery import full_battery
from canonical_tia_core.examples.mock_agents import MockEnvironment, MockInterfaceAgent, MockTaskAgent

def test_battery_runs():
    env = MockEnvironment(); task = MockTaskAgent(); report = full_battery(task, env)
    assert "baseline" in report
    assert "interface_signal" in report

def test_interface_agent_scores_higher_than_task_agent():
    env = MockEnvironment(); task = MockTaskAgent(seed=1); interface = MockInterfaceAgent(seed=1)
    task_report = full_battery(task, env); interface_report = full_battery(interface, env)
    assert interface_report["baseline"]["interface_score"] > task_report["baseline"]["interface_score"]

def test_interface_signal_higher_for_interface_agent():
    env = MockEnvironment(); task = MockTaskAgent(seed=2); interface = MockInterfaceAgent(seed=2)
    task_report = full_battery(task, env); interface_report = full_battery(interface, env)
    assert interface_report["interface_signal"] > task_report["interface_signal"]
