# Observation History

## Purpose

The observer can emit the current observation together with selected observations
from earlier in the same episode. This lets a policy consume temporal context
without adding a separate history node or retaining Arrow payloads downstream.

This feature also aligns arm-position inputs with the canonical OpenArm qpos
schema and propagates the episode attempt identity used by recording and policy
components.

Latency profiling is intentionally outside this change. The node retains the
parallel JPEG decoding already present on `main`, but adds no stage timers,
profiling logs, or profiling-specific CLI options.

## Processing Flow

For each active observer tick, the node:

1. Verifies that all configured arm and camera inputs have been received.
2. Extracts each arm's qpos and concatenates them in right-then-left order.
3. Decodes available JPEG cameras in the persistent worker pool and converts
   them from BGR to RGB.
4. Stores the decoded data as a NumPy-backed observation snapshot.
5. Removes snapshots older than the configured history horizon.
6. Selects the nearest available snapshot for every requested history offset.
7. Constructs one dense Arrow struct whose rows follow the requested offset
   order, then publishes it as `observation`.

Keeping decoded NumPy snapshots avoids retaining nested Arrow observations and
repeatedly concatenating their buffers. Arrow arrays are built only for the
frames selected for the current output.

## History Selection

History offsets are frame indices relative to the newest snapshot. For an
offset `d`, history rate `f`, and newest timestamp `t`, the requested time is:

```text
target_timestamp = t + (d / f) * 1e9
```

Only non-positive offsets are accepted. The snapshot whose timestamp is nearest
to each requested time is selected. For example:

```text
--policy-history-hz 30
--policy-history-delta-indices=-32,0
```

emits the observation nearest to 32 frames in the past followed by the current
observation. During episode startup, when the full horizon is not yet available,
the earliest available snapshot is reused. This keeps the output row count and
schema stable from the first observation onward.

The deque retains the largest requested lookback plus a small 100 ms margin.
History is cleared on episode start, stop, intervention, quit, or stopped-arm
handling, so frames cannot leak between episodes.

## Inputs

The Dora input names are unchanged. The relevant payload requirements are:

| Input | Payload | Behavior |
| --- | --- | --- |
| `arm_right`, `arm_left` | Length-one `struct<qpos: list<float32>>` | Validated and concatenated in right-then-left order. Legacy flat arrays are rejected. |
| Camera inputs | JPEG bytes as `uint8` | Decoded in parallel and emitted as flattened RGB data. |
| `phase_classifier_result` | Length-one Arrow array | Its value and Arrow type are preserved across history rows. |
| `task_prompt` | Length-one string array | Cached in each observation snapshot. Episode metadata may also be updated from this event. |
| `command` | Length-one string array | `start` enables a new episode; `stop`, `intervene`, and `quit` disable output and clear history. |
| `arm_right_status`, `arm_left_status` | Length-one string array | A `stopped` status suppresses output and clears buffered observations. |
| `tick` | Any Dora event | Triggers snapshot construction when all required inputs are ready. |

## Configuration

| CLI option | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `--policy-history-hz` | `POLICY_HISTORY_HZ` | `30.0` | Rate used to convert frame offsets to timestamp offsets. Must be positive. |
| `--policy-history-delta-indices` | `POLICY_HISTORY_DELTA_INDICES` | `0` | Comma-separated, non-positive frame offsets in output row order. |

Existing `--arms` and `--decode-workers` configuration remains available.

## Output Changes

The `observation` output remains a `StructArray` with these fields:

```text
position
camera_wrist_right      # when the right arm is configured
camera_wrist_left       # when the left arm is configured
camera_head_left
camera_head_right
camera_ceiling
phase_classifier_result
task_prompt
id
```

Previously, each output contained one row. It now contains one row per history
offset. With the default offset `0`, the output remains one row. Position and
camera fields use dense `list<float32>` and `list<uint8>` arrays respectively.

Output metadata includes:

| Key | Meaning |
| --- | --- |
| `timestamp` | Timestamp of the newest snapshot in nanoseconds. |
| `episode_number` | Current episode number. |
| `episode_attempt_id` | Current attempt identity when supplied by `start` or `task_prompt`. |
| `history_hz` | Configured history rate. |
| `history_delta_indices` | Requested offsets as a comma-separated string. |
| `history_timestamps` | Actual selected timestamps in output row order. |
| `<camera>.encoding/height/width` | Existing camera format metadata. |

The selected timestamps can repeat during startup padding. The `id` field keeps
the original ID of each selected snapshot, so it can repeat for the same reason.
