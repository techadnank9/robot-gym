from pathvla.schemas import PlanResponseModel, RunRequestModel


def test_plan_schema_validates():
    payload = {
        "subgoals": [
            {
                "type": "navigate",
                "target": "red_bin",
                "constraints": {"avoid": ["chair"], "safe_distance_m": 0.6},
            }
        ]
    }
    plan = PlanResponseModel.model_validate(payload)
    assert plan.subgoals[0].target == "red_bin"


def test_run_request_defaults():
    request = RunRequestModel(instruction="Go to the red bin.")
    assert request.scene == "room"
    assert request.require_vla is True
