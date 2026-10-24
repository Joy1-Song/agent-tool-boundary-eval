"""Decision table of the condition audit. Pure code, no model calls."""

import json
import unittest
from types import SimpleNamespace

from src.condition_audit import Audit, check_call, expected_call, parse_audit, request_audit

META = {"categories": ["登录", "支付", "课程内容"]}


def audit(*conditions):
    return Audit.model_validate({"conditions": list(conditions)})


class ExpectedCallTest(unittest.TestCase):
    def test_count_all_categories(self):
        call, reason = expected_call(audit({"type": "task", "value": "count"}), META)
        self.assertEqual(reason, "ok")
        self.assertEqual(call, {"task_type": "category_overview", "category": None, "date_from": None, "date_to": None})

    def test_unresolved_with_category_and_dates(self):
        call, _ = expected_call(audit(
            {"type": "task", "value": "count"}, {"type": "status", "value": "unresolved"},
            {"type": "category", "value": "付款", "mapped_category": "支付"},
            {"type": "date_range", "date_from": "2026-09-01", "date_to": "2026-09-30"}), META)
        self.assertEqual(call, {"task_type": "unresolved_report", "category": "支付",
                                "date_from": "2026-09-01", "date_to": "2026-09-30"})

    def test_any_unsupported_type_blocks(self):
        for kind in ("relative_time", "channel", "threshold", "exclusion", "group_by", "action", "instruction"):
            call, reason = expected_call(audit({"type": "task", "value": "count"}, {"type": kind, "text": "x"}), META)
            self.assertIsNone(call)
            self.assertTrue(reason.startswith("unsupported"), kind)

    def test_resolved_only_blocks(self):
        self.assertIsNone(expected_call(audit({"type": "task", "value": "count"},
                                              {"type": "status", "value": "resolved"}), META)[0])

    def test_no_task_needs_clarification(self):
        self.assertEqual(expected_call(audit(), META)[1], "clarify:no_task")

    def test_unknown_or_multiple_categories(self):
        self.assertEqual(expected_call(audit({"type": "task", "value": "count"},
                                             {"type": "category", "value": "退款", "mapped_category": None}), META)[1],
                         "clarify:unknown_category")
        self.assertEqual(expected_call(audit({"type": "task", "value": "count"},
                                             {"type": "category", "mapped_category": "支付"},
                                             {"type": "category", "mapped_category": "登录"}), META)[1],
                         "unsupported:multi_category")

    def test_other_or_two_tasks(self):
        self.assertIsNone(expected_call(audit({"type": "task", "value": "other"}), META)[0])
        self.assertIsNone(expected_call(audit({"type": "task", "value": "rating"},
                                              {"type": "task", "value": "unresolved"}), META)[0])


class CheckCallTest(unittest.TestCase):
    def test_match_and_mismatch(self):
        a = audit({"type": "task", "value": "rating"}, {"type": "category", "mapped_category": "登录"})
        good = {"task_type": "rating_analysis", "category": "登录", "date_from": None, "date_to": None}
        self.assertEqual(check_call(a, META, good), (True, "ok"))
        ok, reason = check_call(a, META, {**good, "category": None})
        self.assertFalse(ok)
        self.assertEqual(reason, "mismatch:category")

    def test_planner_refusal_is_kept(self):
        self.assertEqual(check_call(audit({"type": "task", "value": "count"}), META, None), (False, "planner_no_tool"))


class ParseTest(unittest.TestCase):
    def test_malformed_output_raises(self):
        with self.assertRaises(Exception):
            parse_audit("not json")

    def test_unknown_type_and_odd_values_parse_but_fail_closed(self):
        a = parse_audit(json.dumps({"conditions": [{"type": "task", "value": "count"},
                                                   {"type": "rating_filter", "value": 1}]}))
        self.assertEqual(a.conditions[1].type, "unknown")
        self.assertEqual(expected_call(a, META)[1], "unsupported:unknown")
        b = parse_audit(json.dumps({"conditions": [{"type": "task", "value": "unresolved"},
                                                   {"type": "category", "value": ["登录", "支付"]}]}))
        self.assertEqual(expected_call(b, META)[1], "unsupported:multi_category")
        c = parse_audit(json.dumps({"conditions": [{"type": "threshold", "value": 20}]}))
        self.assertEqual(c.conditions[0].value, "20")


class DecisionV2Test(unittest.TestCase):
    """Rules added after the holdout-1 failure analysis (2026-10-24)."""

    def test_group_by_category_is_what_tools_already_do(self):
        call, reason = expected_call(audit({"type": "task", "value": "count"},
                                           {"type": "group_by", "value": "category"}), META)
        self.assertEqual(reason, "ok")
        self.assertEqual(call["task_type"], "category_overview")
        self.assertIsNone(expected_call(audit({"type": "task", "value": "count"},
                                              {"type": "group_by", "value": "channel"}), META)[0])

    def test_unresolved_status_names_the_task(self):
        call, _ = expected_call(audit({"type": "status", "value": "unresolved"},
                                      {"type": "category", "mapped_category": "登录"}), META)
        self.assertEqual(call["task_type"], "unresolved_report")

    def test_all_categories_word_is_not_a_filter(self):
        call, _ = expected_call(audit({"type": "task", "value": "count"},
                                      {"type": "category", "value": "全部", "mapped_category": None}), META)
        self.assertEqual(call["category"], None)

    def test_request_uses_bounded_json_call(self):
        sent = {}

        class Fake:
            def create(self, **payload):
                sent.update(payload)
                message = SimpleNamespace(content=json.dumps({"conditions": [{"type": "task", "value": "count"}]}))
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Fake()))
        result, _ = request_audit(client, "qwen-plus", "按类别统计", META)
        self.assertEqual(result.conditions[0].value, "count")
        self.assertEqual(sent["temperature"], 0)
        self.assertLessEqual(sent["max_tokens"], 500)
        self.assertEqual(sent["response_format"], {"type": "json_object"})

    def test_truncated_response_rejected(self):
        class Fake:
            def create(self, **payload):
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",
                                                                message=SimpleNamespace(content="{"))])

        with self.assertRaises(ValueError):
            request_audit(SimpleNamespace(chat=SimpleNamespace(completions=Fake())), "qwen-plus", "q", META)


if __name__ == "__main__":
    unittest.main()
