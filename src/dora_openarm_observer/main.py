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
import cv2
import dora
import os
import pyarrow as pa
import time


QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])
START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "intervene", "quit"}


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


def _decode_camera(encoded):
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode JPEG camera observation")
    return image


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


def _build_output(
    observation, phase_classifier_result, task_prompt, metadata, decode_pool
):
    """Convert observation to Apache Arrow data and fill metadata.

    observation keys (all values are dora events with a "value" field):
      "arm_right"          – pa.StructArray with qpos, len 1
      "arm_left"           – pa.StructArray with qpos, len 1
      "camera_wrist_right" – JPEG-encoded uint8 flat array, 960×600
      "camera_wrist_left"  – JPEG-encoded uint8 flat array, 960×600
      "camera_head_left"   – JPEG-encoded uint8 flat array, 1280×720
      "camera_head_right"  – JPEG-encoded uint8 flat array, 1280×720
      "camera_ceiling"     – JPEG-encoded uint8 flat array, 960×600
      "id"                 – int64, incremented for each observation

    Output pa.StructArray fields:
      "position"           – concatenated arm positions, list<float32>
      "camera_wrist_right" – decoded RGB flat array, list<uint8>
      "camera_wrist_left"  – decoded RGB flat array, list<uint8>
      "camera_head_left"   – decoded RGB flat array, list<uint8>
      "camera_head_right"  – decoded RGB flat array, list<uint8>
      "camera_ceiling"     – decoded RGB flat array, list<uint8>
      "phase_classifier_result" – StructArray or null
      "task_prompt"        – string (language instruction for the policy)
      "id"                 – int64, incremented for each observation

    metadata is mutated to add per-camera height/width/encoding keys.
    """
    arrays = []
    names = []
    position_arrays = []
    if "arm_right" in observation:
        position_arrays.append(_qpos(observation["arm_right"]["value"]))
    if "arm_left" in observation:
        position_arrays.append(_qpos(observation["arm_left"]["value"]))
    arrays.append(
        pa.array([pa.concat_arrays(position_arrays)], type=pa.list_(pa.float32()))
    )
    names.append("position")

    camera_names = []
    if "camera_wrist_right" in observation:
        camera_names.append("camera_wrist_right")
    if "camera_wrist_left" in observation:
        camera_names.append("camera_wrist_left")
    camera_names.extend(["camera_head_left", "camera_head_right", "camera_ceiling"])
    decode_futures = {
        name: decode_pool.submit(
            _decode_camera,
            observation[name]["value"].to_numpy(),
        )
        for name in camera_names
    }

    def add_camera_observation(name):
        image = decode_futures[name].result()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metadata[f"{name}.encoding"] = "rgb8"
        metadata[f"{name}.height"] = image.shape[0]
        metadata[f"{name}.width"] = image.shape[1]
        arrays.append(pa.array([image.ravel()], type=pa.list_(pa.uint8())))
        names.append(name)

    for name in camera_names:
        add_camera_observation(name)
    if phase_classifier_result is None:
        arrays.append(pa.array([None]))
    else:
        arrays.append(phase_classifier_result)
    names.append("phase_classifier_result")
    arrays.append(pa.array([task_prompt], type=pa.string()))
    names.append("task_prompt")
    arrays.append(pa.array([observation["id"]], type=pa.int64()))
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


def _select_history(history, latest_timestamp, delta_indices, history_hz):
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
    return pa.concat_arrays(selected), selected_timestamps


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
        "--decode-workers",
        default=int(os.getenv("OBSERVER_DECODE_WORKERS", "4")),
        type=int,
        help="Number of persistent JPEG decode workers (default: 4)",
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
    cv2.setNumThreads(1)
    decode_pool = ThreadPoolExecutor(
        max_workers=args.decode_workers,
        thread_name_prefix="observer-jpeg",
    )
    policy_history = deque()

    episode_active = False
    episode_number = 0
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
                arrow_observation = _build_output(
                    observation,
                    last_phase_classifier_result,
                    last_task_prompt,
                    metadata,
                    decode_pool,
                )
                policy_history.append((timestamp, arrow_observation))
                while (
                    policy_history
                    and policy_history[0][0] < timestamp - history_window_ns
                ):
                    policy_history.popleft()

                output, history_timestamps = _select_history(
                    policy_history,
                    timestamp,
                    delta_indices,
                    args.policy_history_hz,
                )
                metadata["history_timestamps"] = ",".join(map(str, history_timestamps))
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
                    policy_history.clear()
                    _reset_observation(observation, arms)
                elif command in STOP_COMMANDS:
                    episode_active = False
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
            else:
                observation[event_id] = event
    finally:
        decode_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
