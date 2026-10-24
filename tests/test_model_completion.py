from types import SimpleNamespace as Obj

import pytest

from src.models import QwenModel
from src.schemas import ProjectError
from test_model_adapter import call_message


@pytest.mark.parametrize("reason,message,code", [
    ("length", Obj(content="", tool_calls=None), "MODEL_TRUNCATED"),
    ("length", call_message(), "MODEL_TRUNCATED"),
    ("content_filter", Obj(content="", tool_calls=None), "MODEL_FILTERED"),
])
def test_incomplete_response_is_not_a_valid_plan_or_refusal(settings, reason, message, code):
    response = Obj(choices=[Obj(message=message, finish_reason=reason)])
    client = Obj(chat=Obj(completions=Obj(create=lambda **kwargs: response)))
    with pytest.raises(ProjectError) as error:
        QwenModel(settings, client=client).plan("分类统计", {}, 3)
    assert error.value.info.code == code


def test_missing_response_choices_is_a_model_error(settings):
    client = Obj(chat=Obj(completions=Obj(create=lambda **kwargs: Obj(choices=[]))))
    with pytest.raises(ProjectError) as error:
        QwenModel(settings, client=client).plan("分类统计", {}, 3)
    assert error.value.info.code == "MODEL_OUTPUT"
