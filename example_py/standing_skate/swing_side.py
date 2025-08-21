#!/usr/bin/python
#
# swing_side.py  – soft ramp to stand_pose.npy, hold, then swing FR & RR legs
#
import sys, time, os, math, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ────────── parameters ──────────
POSE_FILE   = 'stand_pose.npy'
RAMP_SEC    = 3.0
HOLD_SEC    = 5.0
HIP_AMP     = 0.55
KNEE_GAIN   = 0.65
FREQ_HZ     = 1.2
KP0, KD0    = 4.0, 0.3     # start
KPH, KDH    = 12.0, 1.0    # hold
KPS, KDS    = 25.0, 3.0    # swing
ROLL_BIAS   = -0.4
SW_FADE_SEC = 1.0
# ───────────────────────────────

if not os.path.isfile(POSE_FILE):
    raise FileNotFoundError(f'{POSE_FILE} not found – run save_stand_pose.py')
POSE = np.load(POSE_FILE).astype(float)
assert POSE.size == 12, "pose file must contain 12 angles"

d = {f'{leg}_{j}': i
     for i,(leg,j) in enumerate([(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}
RIGHT = ('FR','RR')
SWING = ('FR_1','FR_2','RR_1','RR_2')

LOWLEVEL = 0xff;  DT = 0.002
udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)

cmd   = sdk.LowCmd()           # ← fixed
state = sdk.LowState()         # ← fixed
udp.InitCmdData(cmd)

ω        = 2*math.pi*FREQ_HZ
KNEE_AMP = KNEE_GAIN*HIP_AMP
t0       = time.time()

print("► Streaming in low-level. Robot LED should already be cyan/blue.")

while True:
    loop = time.time()
    udp.Recv(); udp.GetRecv(state)

    t = loop - t0
    # 1) soft ramp
    if t < RAMP_SEC:
        α  = t / RAMP_SEC
        kp = KP0 + α*(KPH-KP0)
        kd = KD0 + α*(KDH-KD0)
        for i in range(12):
            q = state.motorState[i].q*(1-α) + POSE[i]*α
            m = cmd.motorCmd[i]; m.q, m.dq = q, 0; m.Kp, m.Kd = kp, kd; m.tau = 0
    # 2) hold
    elif t < RAMP_SEC + HOLD_SEC:
        for i in range(12):
            m = cmd.motorCmd[i]; m.q, m.dq = POSE[i], 0; m.Kp, m.Kd = KPH, KDH; m.tau = 0
    # 3) swing
    else:
        swing_t   = t - (RAMP_SEC + HOLD_SEC)
        scale     = min(swing_t / SW_FADE_SEC, 1.0)
        hip  =  scale * HIP_AMP  * math.sin(ω*swing_t)
        knee = -scale * KNEE_AMP * math.sin(ω*swing_t)
        for name, idx in d.items():
            moving = name in SWING
            q = POSE[idx]
            if moving:
                if   name.endswith('_1'): q += hip
                elif name.endswith('_2'): q += knee
            kp, kd = (KPS, KDS) if moving else (KPH, KDH)
            m = cmd.motorCmd[idx]; m.q, m.dq = q, 0; m.Kp, m.Kd = kp, kd; m.tau = 0

    # inward roll bias
    cmd.motorCmd[d['FR_0']].tau = ROLL_BIAS
    cmd.motorCmd[d['RR_0']].tau = ROLL_BIAS

    if t > 1.0: safe.PowerProtect(cmd, state, 1)
    udp.SetSend(cmd); udp.Send()

    time.sleep(max(0, DT - (time.time() - loop)))
