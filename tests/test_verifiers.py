"""
Test suite for verifier pipelines, focusing on \boxed{} answer extraction.
"""

import pytest
from src.verifier.integer_verify import extract_boxed_int, verify as int_verify
from src.verifier.sympy_verify import _extract_boxed_expr


class TestIntegerVerifier:
    """Test integer verifier's boxed extraction."""

    @pytest.mark.parametrize("text,expected", [
        (r"\boxed{42}", 42),
        (r"Some text \boxed{123} more text", 123),
        (r"\boxed{-5}", -5),
        (r"No boxed here", None),
        (r"\boxed{abc}", None),  # Non-numeric
        (r"\boxed{3.14}", None),  # Float, not int
        (r"\boxed{42} \boxed{43}", 42),  # First one
    ])
    def test_extract_boxed_int(self, text, expected):
        assert extract_boxed_int(text) == expected

    @pytest.mark.parametrize("pred_text,ref_answer,expected", [
        (r"\boxed{42}", 42, True),
        (r"\boxed{42}", "42", True),
        (r"\boxed{42}", 43, False),
        (r"No boxed", 42, False),
        (r"\boxed{abc}", 42, False),
    ])
    def test_verify(self, pred_text, ref_answer, expected):
        assert int_verify(pred_text, ref_answer) == expected


class TestSympyVerifier:
    """Test sympy verifier's boxed extraction."""

    @pytest.mark.parametrize("text,expected", [
        (r"\boxed{x^2 + 1}", "x^2 + 1"),
        (r"No boxed", None),
        (r"\boxed{a} \boxed{b}", "a"),  # First one
        (r"\boxed{  spaced  }", "spaced"),
    ])
    def test_extract_boxed_expr(self, text, expected):
        assert _extract_boxed_expr(text) == expected


class TestMathVerifyHF:
    """Test Math-Verify HF verifier."""

    @pytest.fixture
    def mock_math_verify(self, monkeypatch):
        """Mock math_verify functions for testing."""
        import sys
        from unittest.mock import MagicMock

        mock_mv = MagicMock()
        def mock_parse(text):
            import re
            m = re.search(r"\\boxed\{(.+?)\}", str(text) or "", flags=re.DOTALL)
            return m.group(1).strip() if m else str(text)
        mock_mv.parse.side_effect = mock_parse
        mock_mv.verify.side_effect = lambda gold, pred: str(gold) == str(pred)

        sys.modules['math_verify'] = mock_mv
        yield
        if 'math_verify' in sys.modules:
            del sys.modules['math_verify']

    def test_verify_with_mock(self, mock_math_verify):
        from src.verifier.math_verify_hf import verify as mv_verify
        assert mv_verify(r"\boxed{42}", "42")
        assert not mv_verify(r"\boxed{42}", "43")
        assert not mv_verify("No boxed", "42")