import pytest

from pathvla.errors import PlanningError
from pathvla.sorting_agent import (
    GEMINI_DECISION_JSON_SCHEMA,
    GeminiActionDecision,
    SortableObjectModel,
    SortingWorldStateModel,
    _retry_delay_seconds,
    validate_grounded_action,
)


def test_gemini_wire_schema_avoids_legacy_openapi_conversion():
    assert "additionalProperties" not in GEMINI_DECISION_JSON_SCHEMA
    assert "additional_properties" not in GEMINI_DECISION_JSON_SCHEMA
    assert set(GEMINI_DECISION_JSON_SCHEMA["required"]) == {
        "action",
        "target",
        "destination",
        "rationale",
        "expected_outcome",
    }


def test_retry_delay_is_parsed_from_google_quota_error():
    assert _retry_delay_seconds("Please retry in 7.038175874s") == pytest.approx(7.038175874)
    assert _retry_delay_seconds("{'retryDelay': '12s'}") == 12.0


def make_state(robot_pose=(0.0, 0.0, 0.82), held_object=None, item_status="available"):
    return SortingWorldStateModel(
        robot_pose=list(robot_pose),
        held_object=held_object,
        objects=[
            {
                "name": "red_cube",
                "kind": "item",
                "color": "red",
                "pose": [0.5, 0.0, 0.82],
                "status": item_status,
                "assigned_bin": "red_bin" if item_status == "sorted" else None,
            },
            {
                "name": "red_bin",
                "kind": "bin",
                "color": "red",
                "pose": [2.0, 0.0, 0.3],
                "status": "static",
            },
        ],
    )


def test_pick_is_grounded_when_close_and_hand_empty():
    decision = GeminiActionDecision(
        action="pick",
        target="red_cube",
        rationale="The red cube belongs in the red bin.",
        expected_outcome="The right hand holds red_cube.",
    )
    validate_grounded_action(decision, make_state())


def test_pick_is_rejected_when_robot_is_too_far():
    decision = GeminiActionDecision(
        action="pick",
        target="red_cube",
        rationale="Pick the visible item.",
        expected_outcome="The hand holds the item.",
    )
    with pytest.raises(PlanningError, match="navigate closer"):
        validate_grounded_action(decision, make_state(robot_pose=(-2.0, 0.0, 0.82)))


def test_place_requires_matching_held_object_and_bin():
    state = make_state(robot_pose=(2.0, 0.0, 0.82), held_object="red_cube", item_status="held")
    decision = GeminiActionDecision(
        action="place",
        target="red_cube",
        destination="red_bin",
        rationale="The held red cube matches the red bin.",
        expected_outcome="red_cube is inside red_bin.",
    )
    validate_grounded_action(decision, state)


def test_finish_is_rejected_until_every_item_is_sorted():
    decision = GeminiActionDecision(
        action="finish",
        rationale="Sorting is complete.",
        expected_outcome="The episode terminates.",
    )
    with pytest.raises(PlanningError, match="before all items"):
        validate_grounded_action(decision, make_state())
    validate_grounded_action(decision, make_state(item_status="sorted"))


def test_wrong_color_destination_is_rejected():
    state = make_state(robot_pose=(2.0, 0.0, 0.82), held_object="red_cube", item_status="held")
    state.objects.append(
        SortableObjectModel.model_validate(
            {
                "name": "blue_bin",
                "kind": "bin",
                "color": "blue",
                "pose": [2.0, 0.0, 0.3],
                "status": "static",
            }
        )
    )
    decision = GeminiActionDecision(
        action="place",
        target="red_cube",
        destination="blue_bin",
        rationale="Attempt the placement.",
        expected_outcome="The item is placed.",
    )
    with pytest.raises(PlanningError, match="Color mismatch"):
        validate_grounded_action(decision, state)
