"""The condition_check switch on QwenModel.plan (regex | audit | both). Fake client, no network."""

import json
import unittest
from types import SimpleNamespace as Obj

from src.config import Settings
from src.models import QwenModel
from src.schemas import ProjectError

META = {"categories": ["登录", "支付", "课程内容"]}


def tool_response(name, arguments):
    call = Obj(id="c1", function=Obj(name=name, arguments=json.dumps(arguments)))
    return Obj(choices=[Obj(finish_reason="tool_calls", message=Obj(content=None, tool_calls=[call]))],
               usage=Obj(prompt_tokens=10, completion_tokens=5))


def audit_response(conditions, raw=None):
    content = raw if raw is not None else json.dumps({"conditions": conditions}, ensure_ascii=False)
    return Obj(choices=[Obj(finish_reason="stop", message=Obj(content=content, tool_calls=None))], usage=None)


class Client:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.chat = Obj(completions=self)

    def create(self, **payload):
        self.requests.append(payload)
        return self.responses.pop(0)


def model(mode, responses):
    client = Client(responses)
    return QwenModel(Settings(condition_check=mode), client=client), client


class ConditionCheckTest(unittest.TestCase):
    def test_default_is_regex_with_one_request(self):
        self.assertEqual(Settings().condition_check, "regex")
        m, client = model("regex", [tool_response("category_overview", {"category": "支付"})])
        m.plan("统计支付类别数量", META, 3)
        self.assertEqual(len(client.requests), 1)

    def test_audit_passes_consistent_call(self):
        m, client = model("audit", [
            tool_response("unresolved_report", {"category": "支付"}),
            audit_response([{"type": "task", "value": "unresolved"},
                            {"type": "category", "value": "付款", "mapped_category": "支付"}]),
        ])
        reply = m.plan("付款这块还有哪些没处理完", META, 3)
        self.assertEqual(reply.value.intent.category, "支付")
        self.assertEqual(reply.usage["audit_requests"], 1)
        self.assertEqual(len(client.requests), 2)

    def test_audit_blocks_condition_regex_cannot_see(self):
        # "微信上反馈的" slips past the regex guard (it only knows "微信渠道")
        responses = [tool_response("category_overview", {"category": "支付"}),
                     audit_response([{"type": "task", "value": "count"},
                                     {"type": "category", "mapped_category": "支付"},
                                     {"type": "channel", "text": "微信上反馈的"}])]
        m, _ = model("regex", responses[:1])
        m.plan("微信上反馈的支付问题统计一下", META, 3)  # regex alone executes
        m, _ = model("audit", responses)
        with self.assertRaises(ProjectError) as error:
            m.plan("微信上反馈的支付问题统计一下", META, 3)
        self.assertEqual(error.exception.info.code, "UNSUPPORTED")

    def test_mismatch_is_retryable_with_feedback(self):
        m, _ = model("audit", [
            tool_response("category_overview", {}),
            audit_response([{"type": "task", "value": "count"}, {"type": "category", "mapped_category": "登录"}]),
        ])
        with self.assertRaises(ProjectError) as error:
            m.plan("登陆类工单数量", META, 3)
        self.assertEqual(error.exception.info.code, "MODEL_INTENT")
        self.assertTrue(error.exception.info.retryable)
        self.assertIn("category", error.exception.info.message)

    def test_unreadable_audit_fails_closed(self):
        m, _ = model("audit", [tool_response("category_overview", {}), audit_response(None, raw="not json")])
        with self.assertRaises(ProjectError) as error:
            m.plan("按类别统计", META, 3)
        self.assertEqual(error.exception.info.code, "AUDIT_FAILED")

    def test_both_runs_regex_first_and_skips_audit_when_blocked(self):
        m, client = model("both", [tool_response("category_overview", {}), audit_response([])])
        with self.assertRaises(ProjectError):
            m.plan("统计微信渠道的数量", META, 3)
        self.assertEqual(len(client.requests), 1)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            Settings(condition_check="magic")


if __name__ == "__main__":
    unittest.main()
