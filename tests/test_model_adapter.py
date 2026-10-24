import json
from types import SimpleNamespace as Obj

import pytest

from src.models import QwenModel, tool_definitions
from src.schemas import ProjectError


class StubClient:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.requests = []
        self.chat = Obj(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return Obj(choices=[Obj(message=next(self.messages))], usage=Obj(prompt_tokens=12, completion_tokens=8))


def call_message(name="category_overview", arguments="{}", call_id="call_test"):
    return Obj(tool_calls=[Obj(id=call_id, function=Obj(name=name, arguments=arguments))], content=None)


def test_real_adapter_message_contract_without_network(settings):
    client = StubClient([call_message(), Obj(content=json.dumps({
        "observations": [{"text": "登录反馈", "ticket_ids": ["T001"]}], "recommendations": [],
    }), tool_calls=None)])
    backend = QwenModel(settings, client=client)
    planned = backend.plan("分类统计", {"categories": ["登录"]}, 3)
    summary = backend.summarize("分类统计", planned.value, {"evidence": [{"ticket_id": "T001"}], "response_mean": 15}, 3)
    assert summary.value.observations[0].ticket_ids == ["T001"]
    messages = client.requests[-1]["messages"]
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "tool"]
    assert messages[-1]["tool_call_id"] == "call_test"
    assert "response_mean" not in messages[-1]["content"]
    assert client.requests[0]["parallel_tool_calls"] is False
    assert planned.usage == {"prompt_tokens": 12, "completion_tokens": 8}


@pytest.mark.parametrize("message,code", [
    (Obj(content="unsupported", tool_calls=None), "UNSUPPORTED"),
    (call_message(name="delete_database"), "MODEL_TOOL"),
    (call_message(arguments='{"path":"/private"}'), "MODEL_ARGUMENTS"),
    (call_message(arguments="not json"), "MODEL_ARGUMENTS"),
])
def test_bad_provider_calls_are_not_executed(settings, message, code):
    backend = QwenModel(settings, client=StubClient([message]))
    with pytest.raises(ProjectError) as error:
        backend.plan("question", {"categories": []}, 3)
    assert error.value.info.code == code


def test_tool_surface_is_limited():
    definitions = tool_definitions()
    assert len(definitions) == 3
    for item in definitions:
        schema = item["function"]["parameters"]
        assert schema["additionalProperties"] is False
        assert "sql" not in schema["properties"] and "path" not in schema["properties"]
