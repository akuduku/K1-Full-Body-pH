"""K1 servo numbers: Eric Plass's standard config of the k1_* skills, RL_Core rl_core/envs/k1_skill_cfg.py.

The configuration k1_sprint, k1_rolling and the stand-up policies are trained with and that run on the real
K1 (eric.plass, RL_Core 61fc36a, delay 302107a).  Per joint group (`"actuators"` there):

  * armature    hip pitch 0.0478125, hip roll 0.0339552, hip yaw 0.0282528, knee 0.095625,
                ankle pitch 0.0395539 / roll 0.0113011 (Booster's parallel linkage), arms and head 0.001
  * EFFORT      torque ceiling: hip pitch 68, hip roll 76, hip yaw 38.3, knee 112, ankles 38.3, arms 14, head 6 N m
  * curve       full torque up to knee_speed, linear to 0 at speed (MOTOR_CURVE)
  * damping     passive joint damping 0.2 on the ankles, none elsewhere
  * kp / kd     legs 80 / 4, ankles 50 / 5.625 (pitch) 3.007 (roll), arms 10 / 1, head 4 / 0.25
  * delay       command delay 2..14 physics steps of 5 ms (10-70 ms), kept for a step with probability 0.3

Joint order: firmware = MJCF order (head 2, arms 8, legs 12).
"""

from __future__ import annotations

SOURCE = "RL_Core rl_core/envs/k1_skill_cfg.py (eric.plass)"

JOINTS = (
    "Head_Yaw", "Head_Pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)

ANKLE_PASSIVE_DAMPING = 0.2  # N m s/rad, the ankles' "damping"
JOINT_DAMPING = 0.0          # every other joint: no passive damping in k1_skill_cfg

_LEG_ARMATURE = (0.0478125, 0.0339552, 0.0282528, 0.095625, 0.0395539, 0.0113011)
ARMATURE = (0.001, 0.001) + (0.001,) * 8 + _LEG_ARMATURE * 2

EFFORT = (6.0, 6.0) + (14.0,) * 8 + (68.0, 76.0, 38.3, 112.0, 38.3, 38.3) * 2

# (effort N m, speed = no-load speed rad/s, knee_speed rad/s) per motor kind
MOTOR_CURVE = {
    "E6408": (68.0, 14.66, 1.88),    # hip pitch
    "E4315": (76.0, 12.57, 2.62),    # hip roll
    "E4310": (38.3, 17.59, 7.85),    # hip yaw, ankles
    "E6416": (112.0, 12.57, 2.09),   # knee
    "R14": (14.0, 33.51, 5.24),      # arms
    "HT4438": (6.0, 7.85, 7.85),     # head: knee = no-load speed (the DC model takes the straight line)
}

# PD gains (kp, kd) per joint
KP = (4.0, 4.0) + (10.0,) * 8 + (80.0, 80.0, 80.0, 80.0, 50.0, 50.0) * 2
KD = (0.25, 0.25) + (1.0,) * 8 + (4.0, 4.0, 4.0, 4.0, 5.625, 3.007) * 2

# command delay: 2..14 steps of 5 ms, kept for a policy step with probability 0.3
DELAY_MS = (10.0, 70.0)
DELAY_HOLD_PROB = 0.3
