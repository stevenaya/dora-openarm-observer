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

"""Node to collect the last observation."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import cv2
import dora
import numpy as np
import os
import pyarrow as pa
import time


QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])
START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "intervene", "quit"}
CAMERA_ORDER = (
    "camera_wrist_right",
    "camera_wrist_left",
    "camera_head_left",
    "camera_head_right",
    "camera_ceiling",
)


@dataclass(frozen=True)
class _ObservationSnapshot:
    position: np.ndarray
    cameras: dict[str, np.ndarray]
    phase_classifier_result: object
    phase_classifier_type: pa.DataType | None
    task_prompt: str | None
    observation_id: int


def _reset_observation(observation, arms):
    """Initialize/reset observations to None and ID to 0."""
    observation.clear()
    if "right" in arms:
        observation["arm_right"] = None
        observation["camera_wrist_right"] = None
    if "left" in arms:
        observation["arm_left"] = None
        observation["camera_wrist_left"] = None
    observation["camera_head_left"] = None
    observation["camera_head_right"] = None
    observation["camera_ceiling"] = None
    observation["id"] = 0


def _qpos(value):
    """Extract qpos from the canonical length-one OpenArm payload."""
    if value.type != QPOS_TYPE or len(value) != 1:
        raise ValueError(
            "Arm position must be a length-one struct<qpos: list<float32>>"
        )
    qpos = value.field("qpos")[0]
    if not qpos.is_valid:
        raise ValueError("Arm position qpos cannot be null")
    return qpos.values


def _decode_camera(encoded):
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode JPEG camera observation")
    return image


def _build_snapshot(
    observation,
    phase_classifier_result,
    task_prompt,
    decode_pool,
):
    """Decode one observation into a history snapshot.

    observation keys (all values are dora events with a "value" field):
      "arm_right"          – pa.StructArray with qpos, len 1
      "arm_left"           – pa.StructArray with qpos, len 1
      "camera_wrist_right" – JPEG-encoded uint8 flat array, 960×600
      "camera_wrist_left"  – JPEG-encoded uint8 flat array, 960×600
      "camera_head_left"   – JPEG-encoded uint8 flat array, 1280×720
      "camera_head_right"  – JPEG-encoded uint8 flat array, 1280×720
      "camera_ceiling"     – JPEG-encoded uint8 flat array, 960×600
      "id"                 – int64, incremented for each observation

    JPEG decode is submitted to the persistent executor before results are
    consumed in canonical camera order.
    """
    position_arrays = []
    if "arm_right" in observation:
        position_arrays.append(_qpos(observation["arm_right"]["value"]).to_numpy())
    if "arm_left" in observation:
        position_arrays.append(_qpos(observation["arm_left"]["value"]).to_numpy())
    position = np.concatenate(position_arrays).astype(np.float32, copy=False)

    camera_names = tuple(name for name in CAMERA_ORDER if name in observation)
    futures = {
        name: decode_pool.submit(
            _decode_camera,
            observation[name]["value"].to_numpy(),
        )
        for name in camera_names
    }

    cameras = {}
    for name in camera_names:
        image = futures[name].result()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cameras[name] = image

    if phase_classifier_result is None:
        phase_value = None
        phase_type = None
    else:
        if len(phase_classifier_result) != 1:
            raise ValueError("Phase classifier result must contain exactly one row")
        phase_value = phase_classifier_result[0].as_py()
        phase_type = phase_classifier_result.type

    return _ObservationSnapshot(
        position=position,
        cameras=cameras,
        phase_classifier_result=phase_value,
        phase_classifier_type=phase_type,
        task_prompt=task_prompt,
        observation_id=observation["id"],
    )


def _list_array(rows, numpy_dtype, arrow_type):
    """Build a dense Arrow list array with one final values buffer."""
    flattened = [np.asarray(row).reshape(-1) for row in rows]
    row_size = flattened[0].size
    if any(row.size != row_size for row in flattened):
        raise ValueError("History rows must have matching shapes")
    values = np.ascontiguousarray(np.stack(flattened), dtype=numpy_dtype).reshape(-1)
    offsets = np.arange(0, values.size + 1, row_size, dtype=np.int32)
    return pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(values, type=arrow_type),
    )


def _build_output(snapshots, metadata):
    """Construct the final selected history as one Arrow struct array."""
    if not snapshots:
        raise ValueError("Cannot build output from empty history")

    arrays = [
        _list_array(
            [snapshot.position for snapshot in snapshots],
            np.float32,
            pa.float32(),
        )
    ]
    names = ["position"]

    camera_names = tuple(name for name in CAMERA_ORDER if name in snapshots[-1].cameras)
    for name in camera_names:
        images = [snapshot.cameras[name] for snapshot in snapshots]
        arrays.append(_list_array(images, np.uint8, pa.uint8()))
        names.append(name)
        metadata[f"{name}.encoding"] = "rgb8"
        metadata[f"{name}.height"] = images[-1].shape[0]
        metadata[f"{name}.width"] = images[-1].shape[1]

    phase_type = next(
        (
            snapshot.phase_classifier_type
            for snapshot in snapshots
            if snapshot.phase_classifier_type is not None
        ),
        None,
    )
    phase_values = [snapshot.phase_classifier_result for snapshot in snapshots]
    arrays.append(
        pa.nulls(len(snapshots))
        if phase_type is None
        else pa.array(phase_values, type=phase_type)
    )
    names.append("phase_classifier_result")
    arrays.append(
        pa.array([snapshot.task_prompt for snapshot in snapshots], type=pa.string())
    )
    names.append("task_prompt")
    arrays.append(
        pa.array([snapshot.observation_id for snapshot in snapshots], type=pa.int64())
    )
    names.append("id")
    return pa.StructArray.from_arrays(arrays, names)


def _parse_delta_indices(value):
    """Parse non-positive policy history frame offsets."""
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indices:
        raise ValueError("At least one policy history delta index is required")
    if any(index > 0 for index in indices):
        raise ValueError("Policy history delta indices must be non-positive")
    return indices


def _select_history_items(history, latest_timestamp, delta_indices, history_hz):
    """Select nearest frames and pad startup history with this episode's first."""
    if not history:
        raise ValueError("Cannot select from empty policy history")

    selected = []
    selected_timestamps = []
    for delta_index in delta_indices:
        target = latest_timestamp + int(delta_index / history_hz * 1_000_000_000)
        timestamp, observation = min(
            history,
            key=lambda item: abs(item[0] - target),
        )
        selected.append(observation)
        selected_timestamps.append(timestamp)
    return selected, selected_timestamps


def main():
    """Collect the last observation."""
    parser = argparse.ArgumentParser(description="Collect the last observation")
    parser.add_argument(
        "--arms",
        default=os.getenv("ARMS", "right,left"),
        help="The used arms: 'right,left' (default), 'right' or 'left'",
        type=str,
    )
    parser.add_argument(
        "--policy-history-hz",
        default=float(os.getenv("POLICY_HISTORY_HZ", "30.0")),
        type=float,
        help="Sampling rate used to interpret policy history offsets",
    )
    parser.add_argument(
        "--policy-history-delta-indices",
        default=os.getenv("POLICY_HISTORY_DELTA_INDICES", "0"),
        help="Comma-separated policy history offsets, for example '-32,0'",
    )
    parser.add_argument(
        "--decode-workers",
        default=int(os.getenv("OBSERVER_DECODE_WORKERS", "4")),
        type=int,
        help="Persistent JPEG decode workers (default: 4)",
    )
    args = parser.parse_args()

    arms = args.arms.split(",")
    if not arms or any(arm not in {"right", "left"} for arm in arms):
        raise ValueError("--arms must contain only 'right' and/or 'left'")
    if args.policy_history_hz <= 0:
        raise ValueError("--policy-history-hz must be positive")
    if args.decode_workers <= 0:
        raise ValueError("--decode-workers must be positive")

    delta_indices = _parse_delta_indices(args.policy_history_delta_indices)
    history_window_ns = int(
        (abs(min(delta_indices)) / args.policy_history_hz + 0.1) * 1_000_000_000
    )

    node = dora.Node()
    observation = {}
    _reset_observation(observation, arms)
    policy_history = deque()
    cv2.setNumThreads(1)
    decode_pool = ThreadPoolExecutor(
        max_workers=args.decode_workers,
        thread_name_prefix="observer-jpeg",
    )
    episode_active = False
    episode_number = 0
    episode_attempt_id = None
    last_phase_classifier_result = None
    last_task_prompt = None
    last_arm_right_status = None
    last_arm_left_status = None

    try:
        for event in node:
            if event["type"] != "INPUT":
                continue

            # Main process
            event_id = event["id"]
            if event_id == "tick":
                if any(v is None for v in observation.values()):
                    # If any observation isn't ready yet, we skip this tick.
                    continue
                if (
                    not episode_active
                    or ("right" in arms and last_arm_right_status == "stopped")
                    or ("left" in arms and last_arm_left_status == "stopped")
                ):
                    policy_history.clear()
                    _reset_observation(observation, arms)
                    continue

                timestamp = time.time_ns()
                metadata = {
                    "episode_number": episode_number,
                    "timestamp": timestamp,
                    "history_hz": args.policy_history_hz,
                    "history_delta_indices": ",".join(map(str, delta_indices)),
                }
                if episode_attempt_id is not None:
                    metadata["episode_attempt_id"] = episode_attempt_id
                snapshot = _build_snapshot(
                    observation,
                    last_phase_classifier_result,
                    last_task_prompt,
                    decode_pool,
                )
                policy_history.append((timestamp, snapshot))
                while (
                    policy_history
                    and policy_history[0][0] < timestamp - history_window_ns
                ):
                    policy_history.popleft()

                selected_snapshots, history_timestamps = _select_history_items(
                    policy_history,
                    timestamp,
                    delta_indices,
                    args.policy_history_hz,
                )
                metadata["history_timestamps"] = ",".join(map(str, history_timestamps))
                output = _build_output(selected_snapshots, metadata)
                node.send_output(
                    "observation",
                    output,
                    metadata,
                )
                observation["id"] += 1

            elif event_id == "command":
                command = event["value"][0].as_py()
                if command in START_COMMANDS:
                    episode_active = True
                    episode_number = int(
                        event["metadata"].get("episode_number", episode_number)
                    )
                    episode_attempt_id = event["metadata"].get("episode_attempt_id")
                    policy_history.clear()
                    _reset_observation(observation, arms)
                elif command in STOP_COMMANDS:
                    episode_active = False
                    episode_attempt_id = None
                    policy_history.clear()
                    _reset_observation(observation, arms)

            elif event_id == "arm_right_status":
                last_arm_right_status = event["value"][0].as_py()
            elif event_id == "arm_left_status":
                last_arm_left_status = event["value"][0].as_py()
            elif event_id == "phase_classifier_result":
                last_phase_classifier_result = event["value"]
            elif event_id == "task_prompt":
                last_task_prompt = event["value"][0].as_py()
                episode_number = int(
                    event["metadata"].get("episode_number", episode_number)
                )
                episode_attempt_id = event["metadata"].get(
                    "episode_attempt_id", episode_attempt_id
                )
            else:
                observation[event_id] = event
    finally:
        decode_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
