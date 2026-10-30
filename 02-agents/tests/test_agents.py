import pytest
from agent_lab.core import Action, Simulator, UnsafePlan, demo_inventory, deterministic_plan, run_plan, validate_plan


def test_guardrail_blocks_unknown_tools_and_extra_fields():
    inv = demo_inventory(1)
    with pytest.raises(UnsafePlan):
        validate_plan([{"tool":"shell", "service":"service-001", "command":"rm -rf /"}], inv)
    with pytest.raises(UnsafePlan):
        validate_plan([{"tool":"migrate_service", "service":"not-real"}], inv)
    with pytest.raises(UnsafePlan):
        validate_plan([{"tool":"migrate_service", "service":"service-001", "force":True}], inv)


def test_dry_run_has_zero_side_effects():
    inv = demo_inventory(3)
    sim = Simulator(inv)
    result = run_plan(sim, validate_plan(deterministic_plan(inv), inv))
    assert result["counts"]["dry_run"] == 3
    assert all(s.version == 1 for s in inv.values())
    assert not sim.completed


def test_run_40_and_suppress_duplicates():
    inv = demo_inventory(40)
    sim = Simulator(inv)
    actions = validate_plan(deterministic_plan(inv), inv)
    result = run_plan(sim, actions + actions[:2], execute=True, workers=3)
    assert result["counts"] == {"passed":40, "failed":0, "skipped":2, "dry_run":0}
    assert all(s.version == 2 for s in inv.values())
    assert run_plan(sim, actions, execute=True)["counts"]["skipped"] == 40


def test_executor_refuses_unvalidated_action():
    with pytest.raises(UnsafePlan):
        run_plan(Simulator(demo_inventory(1)), [Action("shell", "service-001")], execute=True)


def test_injected_failure_is_reported_without_pretending_success():
    inventory = demo_inventory(3)
    sim = Simulator(inventory, fail_services={"service-002"})
    actions = validate_plan(deterministic_plan(inventory), inventory)
    report = run_plan(sim, actions, execute=True)
    assert report["counts"]["passed"] == 2
    assert report["counts"]["failed"] == 1
    assert inventory["service-002"].version == 1
