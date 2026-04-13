from __future__ import annotations
import json
from pathlib import Path
from canonical_tia_core.benchmarks.interface_battery import full_battery
from canonical_tia_core.examples.mock_agents import MockEnvironment, MockTaskAgent, MockInterfaceAgent

def main():
    env = MockEnvironment()
    task = MockTaskAgent()
    interface = MockInterfaceAgent()
    report = {"task_agent": full_battery(task, env), "interface_agent": full_battery(interface, env)}
    out = Path("mock_benchmark_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {out.resolve()}")

if __name__ == "__main__":
    main()
