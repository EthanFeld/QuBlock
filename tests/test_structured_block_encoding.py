from __future__ import annotations

import numpy as np
import pytest

from blockflow.primitives.block_encoding import BlockEncoding
from blockflow.primitives.linear_operator import DiagonalOperator, PermutationOperator
from blockflow.primitives.resources import ResourceEstimate


def test_diagonal_operator_apply() -> None:
    op = DiagonalOperator(np.array([1.0, -2.0]))
    vec = np.array([3.0, 4.0])
    assert np.allclose(op.apply(vec), np.array([3.0, -8.0]))
    assert np.allclose(op.apply_adjoint(vec), np.array([3.0, -8.0]))
    assert np.isclose(op.norm_bound(), 2.0)
    assert op.dtype == np.array([1.0, -2.0]).dtype


def test_permutation_operator_apply() -> None:
    op = PermutationOperator(np.array([1, 0, 2]))
    vec = np.array([10.0, 20.0, 30.0])
    assert np.allclose(op.apply(vec), np.array([20.0, 10.0, 30.0]))
    assert np.allclose(op.apply_adjoint(vec), np.array([20.0, 10.0, 30.0]))
    assert np.isclose(op.norm_bound(), 1.0)
    assert op.dtype == np.array([1, 0, 2]).dtype


def test_structured_block_encoding_constructors() -> None:
    diag = np.array([1.0, 2.0])
    be = BlockEncoding.from_diagonal(diag, resources=ResourceEstimate())
    out = be.semantic_apply(np.array([3.0, 4.0]))
    assert np.allclose(out, np.array([3.0, 8.0]))

    perm = np.array([1, 0])
    be_perm = BlockEncoding.from_permutation(perm, resources=ResourceEstimate())
    out_perm = be_perm.semantic_apply(np.array([5.0, 6.0]))
    assert np.allclose(out_perm, np.array([6.0, 5.0]))


def test_permutation_operator_validation() -> None:
    with pytest.raises(ValueError, match="permutation"):
        PermutationOperator(np.array([0, 0, 1]))
    with pytest.raises(TypeError, match="integers"):
        PermutationOperator(np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="non-empty"):
        PermutationOperator(np.array([], dtype=int))


def test_diagonal_operator_validation() -> None:
    with pytest.raises(ValueError, match="1D"):
        DiagonalOperator(np.eye(2))
    with pytest.raises(ValueError, match="non-empty"):
        DiagonalOperator(np.array([]))


def test_structured_block_encoding_constructor_errors() -> None:
    with pytest.raises(ValueError, match="1D"):
        BlockEncoding.from_diagonal(np.eye(2), resources=ResourceEstimate())
    with pytest.raises(ValueError, match="nonzero"):
        BlockEncoding.from_diagonal(np.array([]), resources=ResourceEstimate())
