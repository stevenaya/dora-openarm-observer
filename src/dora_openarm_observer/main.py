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
import math
import numpy as np
import os
import pyarrow as pa
import statistics
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


class _TimingWindow:
    """Print aggregate observer timings without logging every frame."""

    STAGES = (
        ("total_ms", "total"),
        ("build_ms", "build"),
        ("qpos_ms", "qpos"),
        ("decode_ms", "decode"),
        ("color_ms", "color"),
        ("camera_arrow_ms", "camera_arrow"),
        ("output_arrow_ms", "other_arrow"),
        ("history_ms", "history"),
        ("select_ms", "select"),
        ("send_ms", "send"),
    )

    def __init__(self, every):
        self.every = every
        self.samples = []

    def reset(self):
        """Discard an incomplete timing window at an episode boundary."""
        self.samples.clear()

    @staticmethod
    def _stats(values):
        ordered = sorted(values)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        return (
            statistics.fmean(ordered),
            statistics.median(ordered),
            ordered[p95_index],
        )

    def add(self, sample):
        if self.every <= 0:
            return
        self.samples.append(sample)
        if len(self.samples) < self.every:
            return

        stage_parts = []
        for key, label in self.STAGES:
            values = [item[key] for item in self.samples if key in item]
            if values:
                mean, p50, p95 = self._stats(values)
                stage_parts.append(f"{label}={mean:.2f}/{p50:.2f}/{p95:.2f}ms")

        periods = [
            item["output_period_ms"]
            for item in self.samples
            if "output_period_ms" in item
        ]
        period_part = ""
        if periods:
            mean, p50, p95 = self._stats(periods)
            period_part = f" output_period={mean:.2f}/{p50:.2f}/{p95:.2f}ms"
        print(
            f"[observer timing] n={len(self.samples)} mean/p50/p95 "
            + " ".join(stage_parts)
            + period_part,
            flush=True,
        )

        camera_names = sorted(
            {
                key.removesuffix("_decode_ms")
                for item in self.samples
                for key in item
                if key.startswith("camera_") and key.endswith("_decode_ms")
            }
        )
        camera_parts = []
        for name in camera_names:
            label = name.removeprefix("camera_")
            stage_values = []
            for stage in ("decode", "color", "arrow"):
                key = f"{name}_{stage}_ms"
                values = [item[key] for item in self.samples if key in item]
                stage_values.append(statistics.fmean(values))
            camera_parts.append(
                f"{label}={stage_values[0]:.2f}/"
                f"{stage_values[1]:.2f}/{stage_values[2]:.2f}ms"
            )
        if camera_parts:
            print(
                "[observer cameras] mean decode/color/arrow " + " ".join(camera_parts),
                flush=True,
            )
        self.samples.clear()


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
    start_ns = time.perf_counter_ns()
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    done_ns = time.perf_counter_ns()
    if image is None:
        raise ValueError("Failed to decode JPEG camera observation")
    return image, (done_ns - start_ns) / 1e6


def _build_snapshot(
    observation,
    phase_classifier_result,
    task_prompt,
    decode_pool,
    timing=None,
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

    JPEG decode is submitted to a persistent executor. Color conversion stays
    on this thread because it is small and keeps decode wall time measurable.
    """
    build_start_ns = time.perf_counter_ns()
    qpos_start_ns = time.perf_counter_ns()
    position_arrays = []
    if "arm_right" in observation:
        position_arrays.append(_qpos(observation["arm_right"]["value"]).to_numpy())
    if "arm_left" in observation:
        position_arrays.append(_qpos(observation["arm_left"]["value"]).to_numpy())
    position = np.concatenate(position_arrays).astype(np.float32, copy=False)
    if timing is not None:
        timing["qpos_ms"] = (time.perf_counter_ns() - qpos_start_ns) / 1e6
        timing["color_ms"] = 0.0
        timing["camera_arrow_ms"] = 0.0

    camera_names = tuple(name for name in CAMERA_ORDER if name in observation)
    decode_start_ns = time.perf_counter_ns()
    futures = {
        name: decode_pool.submit(
            _decode_camera,
            observation[name]["value"].to_numpy(),
        )
        for name in camera_names
    }
    decoded = {}
    for name in camera_names:
        image, decode_ms = futures[name].result()
        decoded[name] = image
        if timing is not None:
            timing[f"{name}_decode_ms"] = decode_ms
    decode_done_ns = time.perf_counter_ns()
    if timing is not None:
        timing["decode_ms"] = (decode_done_ns - decode_start_ns) / 1e6

    cameras = {}
    for name in camera_names:
        color_start_ns = time.perf_counter_ns()
        image = decoded[name]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color_done_ns = time.perf_counter_ns()
        cameras[name] = image
        if timing is not None:
            color_ms = (color_done_ns - color_start_ns) / 1e6
            timing[f"{name}_color_ms"] = color_ms
            timing["color_ms"] += color_ms

    if phase_classifier_result is None:
        phase_value = None
        phase_type = None
    else:
        if len(phase_classifier_result) != 1:
            raise ValueError("Phase classifier result must contain exactly one row")
        phase_value = phase_classifier_result[0].as_py()
        phase_type = phase_classifier_result.type

    if timing is not None:
        timing["build_ms"] = (time.perf_counter_ns() - build_start_ns) / 1e6
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


def _build_output(snapshots, metadata, timing=None):
    """Construct the final selected history as one Arrow struct array."""
    if not snapshots:
        raise ValueError("Cannot build output from empty history")

    output_start_ns = time.perf_counter_ns()
    arrays = [
        _list_array(
            [snapshot.position for snapshot in snapshots],
            np.float32,
            pa.float32(),
        )
    ]
    names = ["position"]

    camera_arrow_ms = 0.0
    camera_names = tuple(name for name in CAMERA_ORDER if name in snapshots[-1].cameras)
    for name in camera_names:
        arrow_start_ns = time.perf_counter_ns()
        images = [snapshot.cameras[name] for snapshot in snapshots]
        arrays.append(_list_array(images, np.uint8, pa.uint8()))
        names.append(name)
        arrow_ms = (time.perf_counter_ns() - arrow_start_ns) / 1e6
        camera_arrow_ms += arrow_ms
        metadata[f"{name}.encoding"] = "rgb8"
        metadata[f"{name}.height"] = images[-1].shape[0]
        metadata[f"{name}.width"] = images[-1].shape[1]
        if timing is not None:
            timing[f"{name}_arrow_ms"] = arrow_ms

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
    output = pa.StructArray.from_arrays(arrays, names)
    if timing is not None:
        total_arrow_ms = (time.perf_counter_ns() - output_start_ns) / 1e6
        timing["camera_arrow_ms"] = camera_arrow_ms
        timing["output_arrow_ms"] = max(0.0, total_arrow_ms - camera_arrow_ms)
    return output


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


def _select_history(history, latest_timestamp, delta_indices, history_hz):
    """Compatibility helper returning concatenated Arrow history."""
    selected, selected_timestamps = _select_history_items(
        history,
        latest_timestamp,
        delta_indices,
        history_hz,
    )
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
        "--timing-every",
        default=int(os.getenv("OBSERVER_TIMING_EVERY", "0")),
        type=int,
        help="Print aggregate timing statistics every N output observations (0 disables)",
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
    if args.timing_every < 0:
        raise ValueError("--timing-every must be non-negative")
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
    timing_window = _TimingWindow(args.timing_every)
    last_output_start_ns = None
    cv2.setNumThreads(1)
    decode_pool = ThreadPoolExecutor(
        max_workers=args.decode_workers,
        thread_name_prefix="observer-jpeg",
    )
    print(
        f"Observer JPEG decode workers: {args.decode_workers}; "
        f"OpenCV threads: {cv2.getNumThreads()}",
        flush=True,
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

                output_start_ns = time.perf_counter_ns()
                timing = {} if args.timing_every > 0 else None
                if timing is not None and last_output_start_ns is not None:
                    timing["output_period_ms"] = (
                        output_start_ns - last_output_start_ns
                    ) / 1e6
                last_output_start_ns = output_start_ns
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
                    timing,
                )
                history_start_ns = time.perf_counter_ns()
                policy_history.append((timestamp, snapshot))
                while (
                    policy_history
                    and policy_history[0][0] < timestamp - history_window_ns
                ):
                    policy_history.popleft()
                history_done_ns = time.perf_counter_ns()

                select_start_ns = history_done_ns
                selected_snapshots, history_timestamps = _select_history_items(
                    policy_history,
                    timestamp,
                    delta_indices,
                    args.policy_history_hz,
                )
                select_done_ns = time.perf_counter_ns()
                metadata["history_timestamps"] = ",".join(map(str, history_timestamps))
                output = _build_output(selected_snapshots, metadata, timing)
                send_start_ns = time.perf_counter_ns()
                node.send_output(
                    "observation",
                    output,
                    metadata,
                )
                send_done_ns = time.perf_counter_ns()
                if timing is not None:
                    timing["history_ms"] = (history_done_ns - history_start_ns) / 1e6
                    timing["select_ms"] = (select_done_ns - select_start_ns) / 1e6
                    timing["send_ms"] = (send_done_ns - send_start_ns) / 1e6
                    timing["total_ms"] = (send_done_ns - output_start_ns) / 1e6
                    timing_window.add(timing)
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
                    timing_window.reset()
                    last_output_start_ns = None
                    _reset_observation(observation, arms)
                elif command in STOP_COMMANDS:
                    episode_active = False
                    episode_attempt_id = None
                    policy_history.clear()
                    timing_window.reset()
                    last_output_start_ns = None
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
