"""Score complete intent contracts separately from routing and provider failures."""


def score_intents(records: list[dict], prediction_key: str = "prediction") -> dict:
    executed = [row for row in records if row["expected"]["action"] == "execute"]
    blocked = [row for row in records if row["expected"]["action"] == "no_tool"]
    exact = sum(row[prediction_key] == row["expected"] for row in records)
    supported_exact = sum(row[prediction_key] == row["expected"] for row in executed)
    unsafe = sum(row[prediction_key].get("action") == "execute" for row in blocked)
    return {
        "cases": len(records), "exact_contract_matches": exact,
        "exact_contract_rate": exact / len(records) if records else None,
        "supported_cases": len(executed), "supported_exact_matches": supported_exact,
        "supported_exact_rate": supported_exact / len(executed) if executed else None,
        "unsupported_cases": len(blocked), "unsupported_executions": unsafe,
        "unsupported_execution_rate": unsafe / len(blocked) if blocked else None,
        "errors": sum(row[prediction_key].get("action") == "error" for row in records),
    }
