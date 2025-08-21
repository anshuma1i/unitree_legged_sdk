#!/usr/bin/python
#
# swing_hind.py
#
#  • Start the robot in LOW-LEVEL (L1+L2+Start), already hanging.
#  • Runs the same 3-s soft ramp → 5-s quiet hold → continuous swing
#    used in swing_side.py, but this time on *both back legs*:
#        RR_1 RR_2  and  RL_1 RL_2
#
#  Pose file stand_pose.npy must be present in the working directory
#  (recorded on the ground with save_stand_pose.py).
#
import sys, time, os, math, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ────────── parameters (same numbers as swing_side) ──────────
POSE_FILE   = 'stand_pose.npy'
RAMP_SEC    = 3.0            # limp → pose
HOLD_SEC    = 5.0            # pose hold
HIP_AMP     = 0.55           # rad  hip-pitch ±
KNEE_GAIN   = 0.65           # knee_amp = KNEE_GAIN * HIP_AMP   (opposite phase)
FREQ_HZ     = 1.2            # sweep frequency
KP0, KD0    = 4.0,  0.3      # soft gains at start
KPH, KDH    = 12.0, 1.0      # holding gains
KPS, KDS    = 25.0, 3.0      # swing gains
SW_FADE_SEC = 1.0            # fade-in for amplitude
ROLL_BIAS   = -0.4           # inward torque on hip-rolls
# ─────────────────────────────────────────────────────────────

if not os.path.isfile(POSE_FILE):
    raise FileNotFoundError(f'{POSE_FILE} missing – run save_stand_pose.py first')
POSE = np.load(POSE_FILE).astype(float)
assert POSE.size == 12, "pose file must contain 12 joint angles"

# joint map
d = {f'{leg}_{j}': i
     for i,(leg,j) in enumerate([(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}

HIND  = ('RR','RL')
SWING = ('RR_1','RR_2','RL_1','RL_2')   # ← back legs hip-pitch & knee

LOWLEVEL = 0xff
DT       = 0.002
udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)

cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

ω        = 2 * math.pi * FREQ_HZ
KNEE_AMP = KNEE_GAIN * HIP_AMP
t0       = time.time()

print("► swing_hind.py streaming – robot must be in LOW-LEVEL and hanging.")

while True:
    loop_t = time.time()
    udp.Recv();  udp.GetRecv(state)

    t = loop_t - t0

    # 1) soft ramp from limp to stand pose
    if t < RAMP_SEC:
        α  = t / RAMP_SEC
        kp = KP0 + α * (KPH - KP0)
        kd = KD0 + α * (KDH - KD0)
        for i in range(12):
            q_target = state.motorState[i].q * (1 - α) + POSE[i] * α
            m = cmd.motorCmd[i]
            m.q, m.dq, m.Kp, m.Kd, m.tau = q_target, 0.0, kp, kd, 0.0

    # 2) hold pose (no swing yet)
    elif t < RAMP_SEC + HOLD_SEC:
        for i in range(12):
            m = cmd.motorCmd[i]
            m.q, m.dq, m.Kp, m.Kd, m.tau = POSE[i], 0.0, KPH, KDH, 0.0

    # 3) swing both rear legs while holding everything else
    else:
        swing_t   = t - (RAMP_SEC + HOLD_SEC)
        amp_scale = min(swing_t / SW_FADE_SEC, 1.0)
        hip  =  amp_scale * HIP_AMP  * math.sin(ω * swing_t)
        knee = -amp_scale * KNEE_AMP * math.sin(ω * swing_t)

        for name, idx in d.items():
            moving = name in SWING
            q      = POSE[idx]
            if moving:
                if   name.endswith('_1'):  q += hip
                elif name.endswith('_2'):  q += knee
            kp, kd = (KPS, KDS) if moving else (KPH, KDH)
            m = cmd.motorCmd[idx]
            m.q, m.dq, m.Kp, m.Kd, m.tau = q, 0.0, kp, kd, 0.0

    # inward roll bias on both rear hip-rolls
    cmd.motorCmd[d['RR_0']].tau = ROLL_BIAS
    cmd.motorCmd[d['RL_0']].tau = ROLL_BIAS

    if t > 1.0:  safe.PowerProtect(cmd, state, 1)
    udp.SetSend(cmd);  udp.Send()

    time.sleep(max(0.0, DT - (time.time() - loop_t)))
