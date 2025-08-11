#!/usr/bin/python
# push_openloop_side.py — single push (right-side legs), harness only
#
# Phases: RAMP -> PREPARE -> PUSH -> RECOVER -> COAST
#   RAMP:    soft blend from limp to stand_pose.npy with low->hold gains
#   PREPARE: small squat + inward hip-roll bias on STANCE legs (left side)
#   PUSH:    sinusoid on FR/RR hip-pitch & knee; others hold the pose
#   RECOVER: undo PREPARE
#   COAST:   hold pose quietly, then stop
#
import sys, time, math, os, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# -------------------- USER SETTINGS (safe starters) --------------------
POSE_FILE     = 'stand_pose.npy'   # you provided this
PUSH_LEGS     = ('FR','RR')        # right side swings
STANCE_LEGS   = ('FL','RL')        # left side braces

# timings (s)
RAMP_SEC      = 3.0
PREP_SEC      = 0.35
PUSH_SEC      = 0.90
RECOVER_SEC   = 0.40
COAST_SEC     = 1.50

# swing primitive
HIP_AMP       = 0.55               # rad (± about pose hip-pitch)
KNEE_GAIN     = 0.65               # knee_amp = KNEE_GAIN * HIP_AMP (opposite sign)
FREQ_HZ       = 1.2
ROLL_BIAS     = -0.40              # Nm on hip-rolls (keeps knees outboard)

# gains
HOLD_KP, HOLD_KD   = 12.0, 1.0     # quiet brace
SWING_KP, SWING_KD = 25.0, 3.0     # moving joints

# safety thresholds
PITCH_LIM_RAD = math.radians(10.0) # abort if |pitch| > 10°
KNEE_MIN_RAD  = -2.40              # abort if knee below this
# ----------------------------------------------------------------------

# joint map FR_0..RL_2
d = {f'{leg}_{j}': i for i, (leg, j) in enumerate([(L,k) for L in ('FR','FL','RR','RL') for k in range(3)])}

# load pose
if not os.path.isfile(POSE_FILE):
    raise FileNotFoundError(f'{POSE_FILE} not found')
POSE = np.load(POSE_FILE).astype(float)
assert POSE.size == 12, "pose must contain 12 joint angles"

LOWLEVEL = 0xff; DT = 0.002
udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd, state = sdk.LowCmd(), sdk.LowState()
udp.InitCmdData(cmd)

def set_joint(i, q, kp, kd, tau=0.0):
    m = cmd.motorCmd[i]; m.q = float(q); m.dq = 0.0; m.Kp = kp; m.Kd = kd; m.tau = tau

def apply_hold_pose(kp, kd, bias_roll=False):
    for name, idx in d.items():
        tau = ROLL_BIAS if (bias_roll and name.endswith('_0')) else 0.0
        set_joint(idx, POSE[idx], kp, kd, tau)

def ramp_to_pose_soft(t):
    # gain ramp: 4/0.3 -> HOLD gains
    a  = min(max(t / RAMP_SEC, 0.0), 1.0)
    kp = 4.0 + a * (HOLD_KP - 4.0)
    kd = 0.3 + a * (HOLD_KD - 0.3)
    for i in range(12):
        q_now = state.motorState[i].q
        q_tar = (1.0 - a) * q_now + a * POSE[i]
        set_joint(i, q_tar, kp, kd, 0.0)

def stance_bias_and_squat(level):
    """Brace stance legs with small squat & inward bias. Others hold pose."""
    level = max(0.0, min(1.0, level))
    SQUAT_KNEE = -0.15 * level            # ~8.6° bend
    # start from hold pose
    apply_hold_pose(HOLD_KP, HOLD_KD)
    # stance legs: add squat + inward roll bias
    for leg in STANCE_LEGS:
        set_joint(d[f'{leg}_0'], POSE[d[f'{leg}_0']], HOLD_KP, HOLD_KD, ROLL_BIAS)
        set_joint(d[f'{leg}_1'], POSE[d[f'{leg}_1']], HOLD_KP, HOLD_KD, 0.0)
        set_joint(d[f'{leg}_2'], POSE[d[f'{leg}_2']] + SQUAT_KNEE, HOLD_KP, HOLD_KD, 0.0)
    # push hip-rolls: light inward bias helps clearance
    for leg in PUSH_LEGS:
        set_joint(d[f'{leg}_0'], POSE[d[f'{leg}_0']], HOLD_KP, HOLD_KD, ROLL_BIAS)

def swing_push_legs(t_push):
    """Sinusoid on PUSH_LEGS hip-pitch & knee; everyone else at pose."""
    w = 2 * math.pi * FREQ_HZ
    hip  =  HIP_AMP            * math.sin(w * t_push)
    knee = -(KNEE_GAIN*HIP_AMP)* math.sin(w * t_push)
    # hold baseline
    apply_hold_pose(HOLD_KP, HOLD_KD)
    # overwrite swing joints + bias roll
    for leg in PUSH_LEGS:
        set_joint(d[f'{leg}_1'], POSE[d[f'{leg}_1']] + hip,  SWING_KP, SWING_KD, 0.0)  # hip-pitch
        set_joint(d[f'{leg}_2'], POSE[d[f'{leg}_2']] + knee, SWING_KP, SWING_KD, 0.0)  # knee
        set_joint(d[f'{leg}_0'], POSE[d[f'{leg}_0']], HOLD_KP, HOLD_KD, ROLL_BIAS)

def hard_abort(reason):
    print(f"[ABORT] {reason}")
    # send a final safe hold (no torque)
    for i in range(12):
        set_joint(i, state.motorState[i].q, 0, 0, 0)
    udp.SetSend(cmd); udp.Send()
    raise SystemExit(1)

# ------------------------------ MAIN ------------------------------
print("► Enter LOW-LEVEL (L1+L2+Start), hang in harness, then run this script.")
print("► Phases: RAMP → PREPARE → PUSH → RECOVER → COAST")

phase = 'RAMP'
t0 = time.time()

while True:
    loop_t = time.time()
    udp.Recv(); udp.GetRecv(state)
    t = loop_t - t0

    # safety checks (pitch & knee range)
    try:
        pitch = float(state.imu.rpy[1])   # usual SDK order: [roll, pitch, yaw]
    except Exception:
        pitch = 0.0
    if abs(pitch) > PITCH_LIM_RAD:
        hard_abort(f"Pitch |{pitch:.3f} rad| > {PITCH_LIM_RAD:.3f} rad")

    for leg in ('FR','FL','RR','RL'):
        knee_idx = d[f'{leg}_2']
        if state.motorState[knee_idx].q < KNEE_MIN_RAD:
            hard_abort(f"{leg}_knee below limit ({state.motorState[knee_idx].q:.2f} rad)")

    # phase logic
    if phase == 'RAMP':
        ramp_to_pose_soft(t)
        if t >= RAMP_SEC:
            phase, t0 = 'PREPARE', time.time()
            print("→ PREPARE")

    elif phase == 'PREPARE':
        lvl = min(1.0, t / PREP_SEC)
        stance_bias_and_squat(lvl)
        if t >= PREP_SEC:
            phase, t0 = 'PUSH', time.time()
            print("→ PUSH")

    elif phase == 'PUSH':
        swing_push_legs(t)
        if t >= PUSH_SEC:
            phase, t0 = 'RECOVER', time.time()
            print("→ RECOVER")

    elif phase == 'RECOVER':
        lvl = 1.0 - min(1.0, t / RECOVER_SEC)
        stance_bias_and_squat(lvl)
        if t >= RECOVER_SEC:
            phase, t0 = 'COAST', time.time()
            print("→ COAST")

    elif phase == 'COAST':
        apply_hold_pose(HOLD_KP, HOLD_KD, bias_roll=True)
        if t >= COAST_SEC:
            print("✓ Done. Stopping.")
            break

    # enable SDK safety after first second to avoid tripping during very soft start
    if (time.time() - (t0 - 0)) > 1.0:
        safe.PowerProtect(cmd, state, 1)

    udp.SetSend(cmd); udp.Send()
    time.sleep(max(0.0, DT - (time.time() - loop_t)))
