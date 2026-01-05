from __future__ import annotations

import numpy as np

from blockflow.semantic.executor import _env_flag
from blockflow.semantic.state import StateVector


def test_env_flag_parsing(monkeypatch) -> None:
    monkeypatch.delenv("FLAG", raising=False)
    assert _env_flag("FLAG", default=True) is True
    assert _env_flag("FLAG", default=False) is False

    for value in ("0", "false", "no", "off"):
        monkeypatch.setenv("FLAG", value)
        assert _env_flag("FLAG", default=True) is False

    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("FLAG", value)
        assert _env_flag("FLAG", default=False) is True

    monkeypatch.setenv("FLAG", "maybe")
    assert _env_flag("FLAG", default=False) is False


def test_statevector_normalize_inplace_failure_path() -> None:
    state = StateVector(np.array([1, 1], dtype=int))
    state.normalize()
    assert np.allclose(state.data, np.array([0.70710678, 0.70710678]))
