import argparse
import json
from agent_lab.core import Simulator, demo_inventory, deterministic_plan, llm_plan, run_plan, validate_plan


def main():
    p = argparse.ArgumentParser(description="Guarded multi-agent migration simulation")
    p.add_argument("--services", type=int, default=40)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--llm-plan", action="store_true", help="ask an LLM to propose the bounded JSON plan")
    p.add_argument("--execute-simulation", action="store_true", help="mutate only IN-MEMORY service state")
    p.add_argument("--fail-service", action="append", default=[], help="inject failure for a named simulated service")
    p.add_argument("--out", default=None, help="write JSON report to file")
    args = p.parse_args()
    inventory = demo_inventory(args.services)
    raw = llm_plan(inventory) if args.llm_plan else deterministic_plan(inventory)
    actions = validate_plan(raw, inventory)
    report = run_plan(Simulator(inventory, fail_services=set(args.fail_service)), actions, execute=args.execute_simulation, workers=args.workers)
    print(json.dumps({key: report[key] for key in ("mode", "workers", "counts")}, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
