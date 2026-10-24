from src.intent_metrics import score_intents


def test_equal_result_count_does_not_hide_wrong_filter():
    expected = {"action": "execute", "task_type": "category_overview", "category": "支付", "date_from": None, "date_to": None}
    result = score_intents([{"expected": expected, "prediction": {**expected, "category": "登录"}}])
    assert result["supported_exact_matches"] == 0


def test_provider_failure_is_not_successful_refusal():
    result = score_intents([{"expected": {"action": "no_tool"}, "prediction": {"action": "error", "code": "TIMEOUT"}}])
    assert result["errors"] == 1 and result["exact_contract_matches"] == 0


def test_no_cases_are_unmeasured_not_perfect():
    result = score_intents([])
    assert result["exact_contract_rate"] is None and result["supported_exact_rate"] is None


def test_rejection_does_not_count_as_success_on_supported_request():
    result = score_intents([{"expected": {"action": "execute", "category": None}, "prediction": {"action": "no_tool"}}])
    assert result["supported_exact_rate"] == 0
