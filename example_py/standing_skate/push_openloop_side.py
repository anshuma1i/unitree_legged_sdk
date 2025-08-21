#!/usr/bin/python
# push_openloop_side_from_recorded_v2.py
# Sequence:
# 1) RAMP→STAND (stand_pose.npy)
# 2) RAMP→RECORDED_SIDE (your recorded side-stand pose)
# 3) HOLD LEFT legs as in recorded pose
# 4) SWING right legs 3x while left legs hold
# 5) BOARD-UP right legs: copy left-leg squatting angles (FL→FR, RL→RR) and ramp them on
# 6) COAST indefinitely holding "all-legs-on-board squat" until Ctrl+C

import sys, time, math, os, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ------------------ USER SETTINGS ------------------
STAND_POSE_FILE = "stand_pose.npy"
SIDE_POSE_FILE  = "recorded_pose_20250812-193133.npy"   # <-- set to your recorded file

# Network (match your working scripts)
LOWLEVEL = 0xff
LOCAL_PORT = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007

DT = 0.002

# Timing
RAMP_STAND_SEC   = 3.0
RAMP_SIDE_SEC    = 2.0
HOLD_LEFT_SEC    = 0.7            # brief brace before swings
SWING_FREQ_HZ    = 1.2
SWING_CYCLES     = 3              # exactly 3 swings
BOARD_UP_SEC     = 1.2            # ramp right legs onto board (copy left angles)
# COAST is indefinite until Ctrl+C

# Swing shape (relative to recorded side pose on right legs)
HIP_AMP          = 0.55           # rad, hip-pitch
KNEE_GAIN        = 0.65           # knee_amp = KNEE_GAIN * HIP_AMP (opposite sign)

# Gains
KP_HOLD_LEFT, KD_HOLD_LEFT   = 16.0, 1.2    # anchor left legs
KP_HOLD_RIGHT, KD_HOLD_RIGHT = 10.0, 1.0    # quiet hold on right (when not swinging)
KP_SWING, KD_SWING           = 25.0, 3.0    # for swinging joints
KP_RAMP, KD_RAMP             = 12.0, 1.0    # ramp/transition gains
# Safety
PITCH_LIM_RAD   = math.radians(10.0)
KNEE_MIN_RAD    = -2.40
# Optional: small hip-roll bias torque (helps keep knees outboard during swing/coast)
ROLL_BIAS_TAU   = -0.35
# ---------------------------------------------------

# Joint mapping like your scripts: (FR, FL, RR, RL) × joints (0..2)
d = {f'{leg}_{j}': i for i,(leg,j) in enumerate([(L,k) for L in ('FR','FL','RR','RL') for k in range(3)])}

RIGHT_LEGS = ('FR','RR')
LEFT_LEGS  = ('FL','RL')

def load_pose(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")
    arr = np.load(path).astype(float)
    assert arr.size == 12, f"{path} must contain 12 joint angles"
    return arr

POSE_STAND = load_pose(STAND_POSE_FILE)
POSE_SIDE  = load_pose(SIDE_POSE_FILE)

# Build "ALL-ON-BOARD squat" pose:
# copy left leg angles from recorded side to the right legs (FL→FR, RL→RR)
POSE_ALLBOARD = POSE_SIDE.copy()
# FR = FL
POSE_ALLBOARD[d['FR_0']] = POSE_SIDE[d['FL_0']]
POSE_ALLBOARD[d['FR_1']] = POSE_SIDE[d['FL_1']]
POSE_ALLBOARD[d['FR_2']] = POSE_SIDE[d['FL_2']]
# RR = RL
POSE_ALLBOARD[d['RR_0']] = POSE_SIDE[d['RL_0']]
POSE_ALLBOARD[d['RR_1']] = POSE_SIDE[d['RL_1']]
POSE_ALLBOARD[d['RR_2']] = POSE_SIDE[d['RL_2']]

# SDK objects
udp  = sdk.UDP(LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd, state = sdk.LowCmd(), sdk.LowState()
udp.InitCmdData(cmd)

def s_curve(a):  # min-jerk 0..1
    return 3*a*a - 2*a*a*a

def set_joint(i, q, kp, kd, tau=0.0):
    m = cmd.motorCmd[i]
    m.q = float(q); m.dq = 0.0; m.tau = float(tau)
    m.Kp = float(kp); m.Kd = float(kd)

def hold_left_right(left_pose, right_pose, bias_roll=False):
    """Hold left legs at left_pose (strong), right legs at right_pose (softer)."""
    # left legs (anchor)
    for L in LEFT_LEGS:
        for j in range(3):
            idx = d[f'{L}_{j}']
            set_joint(idx, left_pose[idx], KP_HOLD_LEFT, KD_HOLD_LEFT,
                      (ROLL_BIAS_TAU if (bias_roll and j == 0) else 0.0))
    # right legs (soft hold)
    for R in RIGHT_LEGS:
        for j in range(3):
            idx = d[f'{R}_{j}']
            set_joint(idx, right_pose[idx], KP_HOLD_RIGHT, KD_HOLD_RIGHT,
                      (ROLL_BIAS_TAU if (bias_roll and j == 0) else 0.0))

def ramp_pose(from_pose, to_pose, Tsec, kp=KP_RAMP, kd=KD_RAMP):
    steps = max(1, int(Tsec/DT))
    for s in range(steps):
        a = s_curve((s+1)/steps)
        for i in range(12):
            q = (1-a)*from_pose[i] + a*to_pose[i]
            set_joint(i, q, kp, kd, 0.0)
        udp.SetSend(cmd); udp.Send()
        time.sleep(DT)

def safety_check():
    try:
        pitch = float(state.imu.rpy[1])
    except Exception:
        pitch = 0.0
    if abs(pitch) > PITCH_LIM_RAD:
        abort(f"Pitch |{pitch:.3f}| > {PITCH_LIM_RAD:.3f} rad")
    # knee lower bounds
    for leg in ('FR','FL','RR','RL'):
        knee = d[f'{leg}_2']
        if state.motorState[knee].q < KNEE_MIN_RAD:
            abort(f"{leg} knee below limit ({state.motorState[knee].q:.2f} rad)")

def abort(msg):
    print(f"[ABORT] {msg}")
    for i in range(12):
        set_joint(i, state.motorState[i].q, 0.0, 0.0, 0.0)
    udp.SetSend(cmd); udp.Send()
    raise SystemExit(1)

# ------------- MAIN -------------
print("► LOW-LEVEL (L1+L2+Start). Harness on. Board placed. Running sequence…")

# warmup & enable safety after ~1 s
t0 = time.time()
for _ in range(10):
    udp.Recv(); udp.GetRecv(state); time.sleep(0.005)
while time.time() - t0 < 1.0:
    udp.Recv(); udp.GetRecv(state)

# 1) RAMP → STAND
udp.Recv(); udp.GetRecv(state)
q_now = np.array([state.motorState[i].q for i in range(12)], dtype=float)
print("→ RAMP to STAND")
ramp_pose(q_now, POSE_STAND, RAMP_STAND_SEC)

# 2) RAMP → RECORDED SIDE
print("→ RAMP to RECORDED SIDE pose")
ramp_pose(POSE_STAND, POSE_SIDE, RAMP_SIDE_SEC)

# 3) HOLD LEFT only (right soft), brief brace
print("→ HOLD LEFT legs (right soft) before swing")
t_hold = time.time()
while time.time() - t_hold < HOLD_LEFT_SEC:
    udp.Recv(); udp.GetRecv(state); safety_check()
    hold_left_right(POSE_SIDE, POSE_SIDE, bias_roll=True)
    udp.SetSend(cmd); udp.Send()
    time.sleep(DT)

# 4) SWING right legs 3× (left lock at recorded)
print(f"→ SWING right legs ({SWING_CYCLES} cycles @ {SWING_FREQ_HZ:.2f} Hz)")
t_start = time.time()
dur = SWING_CYCLES / SWING_FREQ_HZ
while time.time() - t_start < dur:
    udp.Recv(); udp.GetRecv(state); safety_check()

    t = time.time() - t_start
    w = 2.0*math.pi*SWING_FREQ_HZ
    hip  =  HIP_AMP            * math.sin(w * t)   # right hip-pitch offset
    knee = -(KNEE_GAIN*HIP_AMP)* math.sin(w * t)   # right knee offset (opposite phase)

    # baseline hold: left strong at recorded, right soft at recorded
    hold_left_right(POSE_SIDE, POSE_SIDE, bias_roll=True)

    # overwrite swing joints on RIGHT legs
    for R in RIGHT_LEGS:
        set_joint(d[f'{R}_1'], POSE_SIDE[d[f'{R}_1']] + hip,  KP_SWING, KD_SWING, 0.0)  # hip-pitch
        set_joint(d[f'{R}_2'], POSE_SIDE[d[f'{R}_2']] + knee, KP_SWING, KD_SWING, 0.0)  # knee
        # keep hip-roll with a small bias
        set_joint(d[f'{R}_0'], POSE_SIDE[d[f'{R}_0']], KP_HOLD_RIGHT, KD_HOLD_RIGHT, ROLL_BIAS_TAU)

    udp.SetSend(cmd); udp.Send()
    time.sleep(DT)

# 5) BOARD-UP right legs: copy left-leg recorded angles onto the right; ramp smoothly
print("→ BOARD-UP right legs (copy left-leg squat → right legs)")
# We’ll ramp only RIGHT legs from their current to POSE_ALLBOARD, while LEFT stays locked at recorded.
steps = max(1, int(BOARD_UP_SEC/DT))
for s in range(steps):
    a = s_curve((s+1)/steps)
    udp.Recv(); udp.GetRecv(state); safety_check()

    # left legs: hold recorded side
    for L in LEFT_LEGS:
        for j in range(3):
            idx = d[f'{L}_{j}']
            set_joint(idx, POSE_SIDE[idx], KP_HOLD_LEFT, KD_HOLD_LEFT,
                      (ROLL_BIAS_TAU if j==0 else 0.0))

    # right legs: ramp to copied-left angles (POSE_ALLBOARD)
    for R in RIGHT_LEGS:
        for j in range(3):
            idx = d[f'{R}_{j}']
            q_from = state.motorState[idx].q
            q_to   = POSE_ALLBOARD[idx]
            q_cmd  = (1-a)*q_from + a*q_to
            set_joint(idx, q_cmd, KP_RAMP, KD_RAMP,
                      (ROLL_BIAS_TAU if j==0 else 0.0))

    udp.SetSend(cmd); udp.Send()
    time.sleep(DT)

# 6) COAST indefinitely: hold all four legs in board squat until Ctrl+C
print("→ COAST (all four legs squat on board). Press Ctrl+C to stop.")
try:
    while True:
        udp.Recv(); udp.GetRecv(state); safety_check()
        # hold POSE_ALLBOARD everywhere, with slight bias on hip-rolls
        for leg in ('FR','FL','RR','RL'):
            for j in range(3):
                idx = d[f'{leg}_{j}']
                tau = (ROLL_BIAS_TAU if j==0 else 0.0)
                set_joint(idx, POSE_ALLBOARD[idx], KP_HOLD_LEFT, KD_HOLD_LEFT, tau)
        udp.SetSend(cmd); udp.Send()
        time.sleep(DT)
except KeyboardInterrupt:
    print("\n[CTRL-C] Stopping and releasing…")
finally:
    for i in range(12):
        set_joint(i, state.motorState[i].q, 0.0, 0.0, 0.0)
    udp.SetSend(cmd); udp.Send()
    print("Done.")
