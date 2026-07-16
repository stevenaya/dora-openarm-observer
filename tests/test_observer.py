"""Tests for canonical qpos and policy history selection."""

# ruff: noqa: D103

from collections import deque

import pyarrow as pa
import pytest

from dora_openarm_observer.main import (
    QPOS_TYPE,
    _parse_delta_indices,
    _qpos,
    _select_history,
)


def test_qpos_accepts_only_canonical_payload():
    value = pa.array([{"qpos": [1.0, 2.0]}], type=QPOS_TYPE)

    assert _qpos(value).to_pylist() == [1.0, 2.0]


def test_qpos_rejects_legacy_flat_payload():
    with pytest.raises(ValueError, match="struct<qpos"):
        _qpos(pa.array([1.0, 2.0], type=pa.float32()))


def test_history_pads_startup_from_current_episode():
    timestamp = 2_000_000_000
    history = deque([(timestamp, pa.array([7]))])

    output, selected_timestamps = _select_history(
        history,
        latest_timestamp=timestamp,
        delta_indices=(-32, 0),
        history_hz=30.0,
    )

    assert output.to_pylist() == [7, 7]
    assert selected_timestamps == [timestamp, timestamp]


def test_history_selects_nearest_available_frames():
    history = deque(
        [
            (0, pa.array([0])),
            (1_000_000_000, pa.array([1])),
            (2_000_000_000, pa.array([2])),
        ]
    )

    output, selected_timestamps = _select_history(
        history,
        latest_timestamp=2_000_000_000,
        delta_indices=(-30, 0),
        history_hz=30.0,
    )

    assert output.to_pylist() == [1, 2]
    assert selected_timestamps == [1_000_000_000, 2_000_000_000]


def test_future_history_offsets_are_rejected():
    with pytest.raises(ValueError, match="non-positive"):
        _parse_delta_indices("0,1")
