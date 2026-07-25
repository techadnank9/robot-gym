import pytest

from pathvla.errors import VLAEndpointError
from pathvla.plan_validator import validate_plan_dict


def test_validate_plan_dict_success():
    plan = validate_plan_dict(
        {
            "subgoals": [
                {
                    "type": "inspect",
                    "target": "table",
                    "constraints": {"avoid": ["chair"], "safe_distance_m": 0.7},
                }
            ]
        }
    )
    assert plan.subgoals[0].target == "table"


def test_validate_plan_dict_failure():
    with pytest.raises(VLAEndpointError):
        validate_plan_dict({"subgoals": [{"type": "navigate"}]})
