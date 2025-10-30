import importlib

import pytest


def test_get_verifier_mappings_integer_sympy_mathverify():
    # Fresh import to ensure latest symbols
    runner = importlib.import_module("src.runner")

    # Integer maps to integer_verify.verify
    v = runner.get_verifier("integer")
    from src.verifier import integer_verify as integer_v

    assert v is integer_v.verify

    # Sympy maps to sympy_verify.verify (module exists regardless of sympy availability)
    v2 = runner.get_verifier("sympy")
    from src.verifier import sympy_verify as sympy_v

    assert v2 is sympy_v.verify

    # Math-Verify mapping returns function; the function may raise at call time if dependency missing
    v3 = runner.get_verifier("math-verify")
    from src.verifier import math_verify_hf as mv_v

    assert v3 is mv_v.verify
    # Aliases
    assert runner.get_verifier("math_verify") is mv_v.verify
    assert runner.get_verifier("mathverify") is mv_v.verify
    assert runner.get_verifier("mv") is mv_v.verify


def test_get_verifier_unknown_raises():
    runner = importlib.import_module("src.runner")
    with pytest.raises(ValueError):
        runner.get_verifier("unknown")
