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

"""Collect synchronized OpenArm observations for a policy."""

import argparse
import os
import time
from collections import deque

import cv2
import dora
import pyarrow as pa


QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])
START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "intervene", "quit"}


def _reset_observation(observation, arms):
    """Clear inputs so every episode starts from freshly received data."""
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


def _build_output(observation, phase_classifier_result, task_prompt, metadata):
    """Build one policy observation from the latest synchronized inputs."""
    positions = []
    if "arm_right" in observation:
        positions.append(_qpos(observation["arm_right"]["value"]))
    if "arm_left" in observation:
        positions.append(_qpos(observation["arm_left"]["value"]))

    arrays = [
        pa.array(
            [pa.concat_arrays(positions)],
            type=pa.list_(pa.float32()),
        )
    ]
    names = ["position"]

    def add_camera(name):
        camera = observation[name]
        image = cv2.imdecode(camera["value"].to_numpy(), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metadata[f"{name}.encoding"] = "rgb8"
        metadata[f"{name}.height"] = image.shape[0]
        metadata[f"{name}.width"] = image.shape[1]
        arrays.append(pa.array([image.ravel()], type=pa.list_(pa.uint8())))
        names.append(name)

    if "camera_wrist_right" in observation:
        add_camera("camera_wrist_right")
    if "camera_wrist_left" in observation:
        add_camera("camera_wrist_left")
    add_camera("camera_head_left")
    add_camera("camera_head_right")
    add_camera("camera_ceiling")

    arrays.append(
        pa.array([None]) if phase_classifier_result is None else phase_classifier_result
    )
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
    """Collect and publish observations while an episode is active."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        default=os.getenv("ARMS", "right,left"),
        help="Comma-separated arm sides",
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

    delta_indices = _parse_delta_indices(args.policy_history_delta_indices)
    history_window_ns = int(
        (abs(min(delta_indices)) / args.policy_history_hz + 0.1) * 1_000_000_000
    )

    node = dora.Node()
    observation = {}
    _reset_observation(observation, arms)
    policy_history = deque()

    episode_active = False
    episode_number = 0
    inference_trial_id = 0
    arm_status = {"right": None, "left": None}
    last_phase_classifier_result = None
    last_task_prompt = None

    for event in node:
        if event["type"] != "INPUT":
            continue

        event_id = event["id"]
        if event_id == "tick":
            if any(value is None for value in observation.values()):
                continue
            if not episode_active or any(arm_status[arm] == "stopped" for arm in arms):
                policy_history.clear()
                _reset_observation(observation, arms)
                continue

            timestamp = time.time_ns()
            metadata = {
                "episode_number": episode_number,
                "inference_trial_id": inference_trial_id,
                "timestamp": timestamp,
                "history_hz": args.policy_history_hz,
                "history_delta_indices": ",".join(map(str, delta_indices)),
            }
            current = _build_output(
                observation,
                last_phase_classifier_result,
                last_task_prompt,
                metadata,
            )
            policy_history.append((timestamp, current))
            while (
                policy_history and policy_history[0][0] < timestamp - history_window_ns
            ):
                policy_history.popleft()

            output, history_timestamps = _select_history(
                policy_history,
                timestamp,
                delta_indices,
                args.policy_history_hz,
            )
            metadata["history_timestamps"] = ",".join(map(str, history_timestamps))
            node.send_output("observation", output, metadata)
            observation["id"] += 1

        elif event_id == "command":
            command = event["value"][0].as_py()
            if command in START_COMMANDS:
                episode_active = True
                episode_number = int(
                    event["metadata"].get("episode_number", episode_number)
                )
                inference_trial_id += 1
                policy_history.clear()
                _reset_observation(observation, arms)
            elif command in STOP_COMMANDS:
                episode_active = False
                policy_history.clear()
                _reset_observation(observation, arms)

        elif event_id == "arm_right_status":
            arm_status["right"] = event["value"][0].as_py()
        elif event_id == "arm_left_status":
            arm_status["left"] = event["value"][0].as_py()
        elif event_id == "phase_classifier_result":
            last_phase_classifier_result = event["value"]
        elif event_id == "task_prompt":
            last_task_prompt = event["value"][0].as_py()
        else:
            observation[event_id] = event


if __name__ == "__main__":
    main()
