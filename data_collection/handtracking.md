# Hand Tracking Transform: Old vs. New — Analysis Notes

## Unity → ROS coordinate conversion

### Palm position/orientation

Both the old (`ARTeleopInterface-handtracking`) and new (`ARTeleopInterface-handtrackingROS2`)
`PalmRightPosePublisher.cs` use the **identical `Unity2Ros()` call**:

```csharp
// Old
message.pose.position    = GetGeometryPoint(pose.Position.Unity2Ros());
message.pose.orientation = GetGeometryQuaternion(pose.Rotation.Unity2Ros());
Publish(message);  // always publishes, even if not tracked

// New
if (tracked) {
    message.pose.position    = GetGeometryPoint(pose.Position.Unity2Ros());
    message.pose.orientation = GetGeometryQuaternion(pose.Rotation.Unity2Ros());
    Publish(message);  // only publishes when tracked
}
```

The `Unity2Ros` function is byte-for-byte identical in both projects:
```
Position:    ROS.(x, y, z) = Unity.(z, -x, y)
Quaternion:  ROS.(x, y, z, w) = Unity.(-z, x, -y, w)
```

### Finger tip publisher — this is where old and new differ

| | Old `HandPosePublisher.cs` | New `FingerTipPosePublisher.cs` |
|---|---|---|
| Publish target | Custom `holo_hand` message (all 5 tips in one msg) | Separate `PoseStamped` per finger |
| Position formula | `(-y, -x, z)` — custom `ROS_pose()` | `(z, -x, y)` — `Unity2Ros()` |
| Rotation formula | `(y, -x, z, w)` — custom `ROS_quat()` | `(-z, x, -y, w)` — `Unity2Ros()` |

The finger tip **position z-axis** diverges most: the old code kept Unity.z (depth/forward)
as-is for z, while the new code maps Unity.y → ROS.z (height/up) via `Unity2Ros()`.

However, for robot control purposes both old and new gripper scripts subscribed to
`/hololens/thumb/right` and `/hololens/index/right` as `PoseStamped` topics — those come
from `FingerTipPosePublisher` in both projects, both using `Unity2Ros()`. The custom
`holo_hand` message was a separate data path not used for robot control.

---

## Robot control side — the fundamental difference

### Old approach (`send_cartesian_velocity.py`, ROS1)

Delta-based velocity control: velocity is computed from frame-to-frame palm position change.

```python
# No workspace mapping needed — just differentiate position
dx = curr_pose.position.x - prev_wrist_pose.position.x
vx = dx / dt
twist.linear_x = vx

# Full quaternion delta as rotation vector, but only z used (and negated)
q_delta = q2 * q1.inv()
angular_velocity = q_delta.as_rotvec() / dt
twist.angular_z = -angular_velocity[2]   # negated z only, x/y axes dropped
twist.duration = 0                        # no watchdog
```

### New approach (`kinova_hand_controller.py`, ROS2)

Absolute position P-loop: drives TCP to an absolute target position in robot frame.

```python
# Requires calibrated workspace offsets to map HoloLens world → robot frame
robot_pos = holo_pos * scale + offset
pos_error = target_position - current_pos
raw_vel = p_gain * pos_error

# Absolute yaw delta from a captured reference pose
delta_yaw = palm_yaw - ref_yaw_at_enable
target_theta_z = ref_robot_theta_z + delta_yaw
ang_vel_z = p_gain * (target_theta_z - current_theta_z)
cmd.duration = 200   # watchdog: robot auto-stops if node dies
```

---

## Why this matters for known bugs

| Problem | Old code | New code |
|---|---|---|
| **z doesn't move** | Never an issue — `vz = dz/dt` is frame-agnostic; moves however much hand moves | Requires correct `workspace_z_offset`. If wrong, absolute target is always clipped to workspace floor. Calibrate: `offset = 0.107 − holo_z_at_neutral` |
| **Wrist jump on start** | Never an issue — delta from a wrong reference = zero velocity | Reference captured at enable time. Identity quaternion on first MRTK frame → wrong reference → immediate wrist rotation |
| **Wrist rotation fault** | Dropped angular x/y entirely; only sent negated z. Hard to drive wrist into joint limit | P-loop chases absolute target → sustained max angular velocity if reference was wrong → joint limit fault after ~38 s |
| **Safety watchdog** | `duration=0` — robot keeps moving if node dies | `duration=200 ms` — robot auto-stops |

The old delta approach was **naturally safe for the first-frame identity problem**: a wrong
reference at zero still produces zero delta velocity. The new absolute-position approach is
more precise but requires the workspace offsets and reference capture to be correct.

---

## Fixes applied (this branch)

### 1. Relative wrist yaw tracking with 0.5 s settling period
**File:** `data_collection/kinova_hand_controller.py`

When the Arm button is pressed, the robot stays completely still for 0.5 s while MRTK
tracking stabilises (MRTK can return `tracked=True` with identity rotation for 1–2 frames
on first acquisition). After settling, the controller snapshots the actual palm yaw and the
robot's current `theta_z`. All subsequent wrist rotation is tracked as a delta from that
reference so the robot starts from its current position.

The reference is reset (and re-settled) whenever:
- Arm tracking is disabled
- Hand tracking is lost

Log line confirming capture:
```
Wrist reference captured (after settling): palm_yaw=X°  robot_theta_z=Y°  holo_pos=(x, y, z)
```
If `palm_yaw` is exactly `0.0°`, identity was still caught — try re-pressing Arm.

### 2. Fault detection and auto-recovery
**File:** `data_collection/kinova_hand_controller.py`

When `ROBOT_IN_FAULT` is detected (from control loop or gripper command):
- All motion commands are immediately suppressed (stops 30 Hz command spam)
- `arm_enabled` is cleared — requires explicit re-press of Arm button
- `ClearFaults()` + servoing re-entry is retried every 3 s automatically

Log sequence:
```
ROBOT FAULT DETECTED — arm motion disabled. Auto-clear will be attempted every 3 s.
Robot fault cleared — re-entering servoing mode. Re-press Arm to resume.
```

### 3. Workspace z-offset calibration guidance
**File:** `data_collection/README.md`

With the new absolute-position control, `workspace_z_offset` must be calibrated so your
hand's natural height maps into the robot workspace [0.025, 0.30 m]:

```bash
# Robot disabled — just watch the topic
ros2 topic echo /hololens/palm/right
# Note position.z at your neutral hand height
# Then set:
# workspace_z_offset = home_z (0.107) - holo_z_at_neutral
```

Move hand up/down and confirm `position.z` changes (not `position.x`).

---

## Outstanding items

- **Unity scene bug**: A `PoseStampedPublisher` component in the scene has no
  `PublishedTransform` assigned, causing ~50 NullReferenceExceptions/s in the Unity log.
  In the Unity editor: search the scene hierarchy for objects with a `PoseStampedPublisher`
  component and either assign the transform or remove the component.

- **Angular z negation**: The old `send_cartesian_velocity.py` used
  `angular_z = -angular_velocity[2]` (negated). This may mean the ROS→robot wrist yaw
  direction is inverted. If rotating your wrist clockwise rotates the robot
  counter-clockwise (or vice versa), add `fixed_theta_z_offset_deg` negation or negate
  `delta_yaw` in `kinova_hand_controller.py`.

- **Workspace calibration**: x-range is only 10 cm [0.40, 0.50 m]. Consider widening to
  give more forward/backward range. Full calibration procedure is in the README.
