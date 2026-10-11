"""Tests for task_validation.validator.

The BLOCK cases are modelled on real broken tasks found in the corpus:
an environment holding `eqp_42` while task and ground truth target
`eqp_101`; ground truth that deletes a product and then resolves
conflicts on it; ground truth longer than the turn budget.
"""

from task_validation import BLOCK, WARN, validate_task
from task_validation.validator import entity_ids, parse_call, tool_names


def _row(**over):
    row = {
        "id": "demo_spec_001_seq1",
        "summary": "Collect sensor data for equipment eqp_42 and detect anomalies.",
        "tools": [{"name": "CollectSensorData"}, {"name": "DetectAnomalies"}],
        "gt_tool_calls": [
            "CollectSensorData(equipment_id='eqp_42', window_minutes=60)",
            "DetectAnomalies(equipment_id='eqp_42', threshold=0.8)",
        ],
        "initial_state": {"equipment": [{"equipment_id": "eqp_42", "name": "Pump"}]},
        "final_state": {"anomalies": []},
    }
    row.update(over)
    return row


# --- helpers ---------------------------------------------------------------

def test_parse_call_roundtrip():
    assert parse_call("Tool(a=1, b='x')") == ("Tool", {"a": 1, "b": "x"})


def test_parse_call_rejects_non_call():
    assert parse_call("not a call") is None
    assert parse_call("") is None
    assert parse_call("Tool(a=1") is None


def test_parse_call_keeps_name_when_arg_not_literal():
    name, args = parse_call("Tool(a=some_var)")
    assert name == "Tool"
    assert args["a"] == "some_var"


def test_tool_names_handles_shapes():
    assert tool_names(["A", {"name": "B"}, {"function": {"name": "C"}}]) == {"A", "B", "C"}


def test_entity_ids_picks_handles_not_prose():
    got = entity_ids({"a": "eqp_101 and prod_12345", "b": "just words", "c": "2026-04-23"})
    assert "eqp_101" in got and "prod_12345" in got
    assert not any(g in {"just", "words"} for g in got)


# --- clean task ------------------------------------------------------------

def test_clean_task_not_blocked():
    rep = validate_task(_row(), max_solver_turns=20)
    assert not rep.blocked, rep.codes


# --- V1..V4 ----------------------------------------------------------------

def test_v1_empty_gt_blocks():
    rep = validate_task(_row(gt_tool_calls=[]))
    assert rep.blocked and "V1_gt_empty" in rep.codes


def test_v2_unparseable_gt_warns_but_does_not_block():
    """A parse failure is our parser's limit, not proof the task is broken."""
    rep = validate_task(_row(gt_tool_calls=["CollectSensorData(equipment_id='eqp_42'"]))
    assert not rep.blocked
    assert "V2_gt_unparseable" in rep.codes


def test_v3_unknown_tool_blocks():
    rep = validate_task(_row(gt_tool_calls=["GhostTool(equipment_id='eqp_42')"]))
    assert rep.blocked and "V3_unknown_tool_in_gt" in rep.codes


def test_v4_budget_infeasible_blocks():
    rep = validate_task(_row(), max_solver_turns=1)
    assert rep.blocked and "V4_budget_infeasible" in rep.codes


def test_v4_silent_without_budget():
    rep = validate_task(_row())
    assert "V4_budget_infeasible" not in rep.codes


# --- V5: the canonical broken-task family ----------------------------------

def test_v5_entity_missing_from_environment_blocks():
    """Task and GT target eqp_101; the environment only holds eqp_42."""
    rep = validate_task(_row(
        summary="Collect sensor data for equipment eqp_101 and detect anomalies.",
        gt_tool_calls=[
            "CollectSensorData(equipment_id='eqp_101', window_minutes=60)",
            "DetectAnomalies(equipment_id='eqp_101', threshold=0.8)",
        ],
    ), max_solver_turns=20)
    assert rep.blocked
    assert "V5_entity_not_in_env" in rep.codes


def test_v5_allows_id_introduced_by_earlier_gt_call():
    """An id first seen in a later call may have come from an earlier response."""
    rep = validate_task(_row(
        tools=[{"name": "CollectSensorData"}, {"name": "DetectAnomalies"},
               {"name": "FileReport"}],
        gt_tool_calls=[
            "CollectSensorData(equipment_id='eqp_42', window_minutes=60)",
            "DetectAnomalies(equipment_id='eqp_42', threshold=0.8)",
            "FileReport(anomaly_id='anom_77', equipment_id='eqp_42')",
        ],
    ), max_solver_turns=20)
    assert "V5_entity_not_in_env" not in rep.codes


def test_v5_ignores_minted_id_when_env_has_no_such_family():
    """`OFFERING_001` is a label the task assigns, not a lookup: env has no
    offering-shaped id to contradict it, so absence is not provable."""
    rep = validate_task(_row(
        tools=[{"name": "CollectSensorData"}, {"name": "DetectAnomalies"},
               {"name": "DecisionLogRecorder"}],
        summary="Collect sensor data for eqp_42, then log against OFFERING_001.",
        gt_tool_calls=[
            "CollectSensorData(equipment_id='eqp_42', window_minutes=60)",
            "DecisionLogRecorder(offering_id='OFFERING_001', detail='x')",
        ],
    ), max_solver_turns=20)
    assert not rep.blocked, rep.codes


def test_v5_quiet_when_no_initial_state():
    """Without an environment we cannot prove absence; stay silent."""
    rep = validate_task(_row(
        initial_state=None,
        summary="Collect sensor data for equipment eqp_101.",
        gt_tool_calls=["CollectSensorData(equipment_id='eqp_101', window_minutes=60)"],
    ))
    assert "V5_entity_not_in_env" not in rep.codes


# --- V6: self-contradictory ground truth -----------------------------------

def test_v6_use_after_delete_blocks():
    rep = validate_task(_row(
        tools=[{"name": "DeleteProduct"}, {"name": "ConflictResolver"}],
        summary="Delete product prod_12345 then resolve its attribute conflicts.",
        gt_tool_calls=[
            "DeleteProduct(product_id='prod_12345')",
            "ConflictResolver(product_id='prod_12345', strategy='overwrite')",
        ],
        initial_state={"products": [{"product_id": "prod_12345"}]},
    ), max_solver_turns=20)
    assert rep.blocked and "V6_gt_use_after_delete" in rep.codes


def test_v6_ignores_unrelated_entity():
    rep = validate_task(_row(
        tools=[{"name": "DeleteProduct"}, {"name": "ConflictResolver"}],
        summary="Delete prod_12345 then fix prod_99999.",
        gt_tool_calls=[
            "DeleteProduct(product_id='prod_12345')",
            "ConflictResolver(product_id='prod_99999', strategy='overwrite')",
        ],
        initial_state={"products": [{"product_id": "prod_12345"},
                                    {"product_id": "prod_99999"}]},
    ), max_solver_turns=20)
    assert "V6_gt_use_after_delete" not in rep.codes


# --- V7..V9 warn only ------------------------------------------------------

def test_v7_orphan_summary_entity_warns_only():
    rep = validate_task(_row(
        summary="Collect sensor data for eqp_42, then notify ticket_555.",
    ), max_solver_turns=20)
    assert not rep.blocked
    assert "V7_summary_entity_unknown" in rep.codes


def test_v8_trivial_warns():
    rep = validate_task(_row(
        gt_tool_calls=["CollectSensorData(equipment_id='eqp_42', window_minutes=60)"],
    ), max_solver_turns=20)
    assert not rep.blocked and "V8_trivial" in rep.codes


def test_v9_missing_final_state_warns():
    rep = validate_task(_row(final_state=None), max_solver_turns=20)
    assert not rep.blocked and "V9_no_final_state" in rep.codes


def test_report_serialises():
    d = validate_task(_row(), max_solver_turns=20).to_dict()
    assert d["task_id"] == "demo_spec_001_seq1"
    assert "blocked" in d and isinstance(d["findings"], list)
