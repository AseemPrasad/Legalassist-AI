from unittest.mock import MagicMock
from core.precedent_matcher import PrecedentMatcher


def test_get_argument_success_rate_fuzzy():
    # Mock Database Session and Query results
    db = MagicMock()
    
    mock_arg1 = MagicMock()
    mock_arg1.argument_text = "The contract was signed under extreme duress and coercion."
    mock_arg1.argument_succeeded = True
    
    mock_arg2 = MagicMock()
    mock_arg2.argument_text = "Contract execution was forced under duress."
    mock_arg2.argument_succeeded = False
    
    mock_arg3 = MagicMock()
    mock_arg3.argument_text = "Unrelated argument about property boundary."
    mock_arg3.argument_succeeded = True
    
    db.query().all.return_value = [mock_arg1, mock_arg2, mock_arg3]
    
    # We query with a fuzzy variant of duress argument
    res = PrecedentMatcher.get_argument_success_rate(db, "contract signed under duress")
    
    # Total matched should be 2 (mock_arg1 and mock_arg2 match, mock_arg3 is filtered out)
    assert res["total_uses"] == 2
    assert res["successful"] == 1
    assert res["failed"] == 1
    assert res["success_rate"] == 50.0
