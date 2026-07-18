# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for parallel decoding, canonical qpos, and history selection."""

# ruff: noqa: D103

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_observer.main import (
    _ObservationSnapshot,
    _build_output,
    _build_snapshot,
    QPOS_TYPE,
    _parse_delta_indices,
    _qpos,
    _select_history,
)


def _camera_event(value):
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", image)
    assert encoded
    return {"value": pa.array(jpeg, type=pa.uint8())}


def _qpos_event(values):
    return {"value": pa.array([{"qpos": values}], type=QPOS_TYPE)}


def test_parallel_decode_preserves_mainline_output_format():
    observation = {
        "arm_right": _qpos_event([1.0, 2.0]),
        "arm_left": _qpos_event([3.0, 4.0]),
        "camera_wrist_right": _camera_event(10),
        "camera_wrist_left": _camera_event(20),
        "camera_head_left": _camera_event(30),
        "camera_head_right": _camera_event(40),
        "camera_ceiling": _camera_event(50),
        "id": 7,
    }
    metadata = {}

    with ThreadPoolExecutor(max_workers=3) as decode_pool:
        snapshot = _build_snapshot(observation, None, "pick", decode_pool)
    output = _build_output([snapshot], metadata)

    assert output.type.names == [
        "position",
        "camera_wrist_right",
        "camera_wrist_left",
        "camera_head_left",
        "camera_head_right",
        "camera_ceiling",
        "phase_classifier_result",
        "task_prompt",
        "id",
    ]
    assert output.field("position").to_pylist() == [[1.0, 2.0, 3.0, 4.0]]
    assert output.field("position").type == pa.list_(pa.float32())
    assert output.field("camera_ceiling").type == pa.list_(pa.uint8())
    assert output.field("task_prompt").to_pylist() == ["pick"]
    assert output.field("id").to_pylist() == [7]
    assert metadata["camera_ceiling.encoding"] == "rgb8"
    assert metadata["camera_ceiling.height"] == 2
    assert metadata["camera_ceiling.width"] == 3


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


def test_selected_snapshots_build_one_dense_history_output():
    snapshots = [
        _ObservationSnapshot(
            position=np.array([index, index + 0.5], dtype=np.float32),
            cameras={"camera_ceiling": np.full((2, 3, 3), index, dtype=np.uint8)},
            phase_classifier_result=None,
            phase_classifier_type=None,
            task_prompt=f"task-{index}",
            observation_id=index,
        )
        for index in (1, 2)
    ]
    metadata = {}

    output = _build_output(snapshots, metadata)

    assert output.field("position").to_pylist() == [[1.0, 1.5], [2.0, 2.5]]
    camera = output.field("camera_ceiling")
    assert camera.offsets.to_pylist() == [0, 18, 36]
    assert camera[0].values.to_pylist() == [1] * 18
    assert camera[1].values.to_pylist() == [2] * 18
    assert output.field("task_prompt").to_pylist() == ["task-1", "task-2"]
    assert output.field("id").to_pylist() == [1, 2]
    assert metadata == {
        "camera_ceiling.encoding": "rgb8",
        "camera_ceiling.height": 2,
        "camera_ceiling.width": 3,
    }
