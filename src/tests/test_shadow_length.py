"""
Shadow length heuristic tests.
"""
from __future__ import annotations

from src.core.logic.verify import _shadow_length_ok


def test_shadow_length_reasonable() -> None:
    # Elevation 45Â° -> expected ratio around 1.0
    assert _shadow_length_ok(45.0, 1.0)
    assert _shadow_length_ok(45.0, 2.5)
    assert not _shadow_length_ok(45.0, 10.0)


