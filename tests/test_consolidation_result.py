from mnemosyne.consolidator import ConsolidationResult


def test_consolidation_result_defaults():
    result = ConsolidationResult()
    assert result.episodes_processed == 0
    assert result.facts_created == 0
    assert result.errors == []


def test_consolidation_result_with_errors():
    result = ConsolidationResult(errors=["something failed"])
    assert len(result.errors) == 1
