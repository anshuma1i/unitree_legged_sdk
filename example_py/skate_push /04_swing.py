#!/usr/bin/python
#
# 05_swing_side_softlock.py
#
# • Capture full posture, but hold *only FL & RL* with low stiffness
#   to reduce motor noise.
# • FR and RR hip-pitch & knee do the push-style sinusoid.
# -----------------------------------------------------------------

import sys, time, math, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk                                # SDK binding

# ─── motion tunables ─────────────────────────────────────────────
HIP_AMP     = 0.55            # rad  front/back sweep   (right legs)
KNEE_GAIN   = 0.65            # knee_amp = KNEE_GAIN * HIP_AMP
FREQ_HZ     = 1.2             # Hz  (cycles per second)

SWING_KP    = 25              # PD gains for moving joints
SWING_KD    =  3
SOFT_KP     = 10              # low-noise brace for left legs & hip-rolls
SOFT_KD     =  0.5
# ─────────────────────────────────────────────────────────────────

d = {f'{leg}_{j}': i
     for i,(leg,j) in enumerate([(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}

LOWLEVEL = 0xff
DT       = 0.002
PosStopF = 1e9;  VelStopF = 16000.0

# “neutral” centre for roll / pitch / knee (same as original demo)
MID = np.array([0.0, 1.2, -2.0])

RIGHT  = ('FR','RR')
LEFT   = ('FL','RL')
SWING  = ('FR_1','FR_2','RR_1','RR_2')          # hip-pitch & knee only

udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd, state = sdk.LowCmd(), sdk.LowState();  udp.InitCmdData(cmd)

qInit = np.zeros(12);  qDes = np.zeros(12)
motion = ramp_i = sin_i = 0
ω = 2*math.pi*FREQ_HZ
KNEE_AMP = KNEE_GAIN * HIP_AMP

def lerp(a,b,r): r=max(0,min(1,r)); return a*(1-r)+b*r

while True:
    time.sleep(DT);  motion += 1
    udp.Recv(); udp.GetRecv(state)

    # Phase 1: snapshot *all* 12 angles (0-9)
    if motion < 10:
        for name,idx in d.items():
            qInit[idx] = state.motorState[idx].q
        qDes[:] = qInit                                   # initial echo

    # Phase 2: glide *right-side* to MID, leave left at captured (10-399)
    elif motion < 400:
        ramp_i += 1; r = ramp_i/200.0
        for leg in RIGHT:
            for j in range(3):
                idx = d[f'{leg}_{j}']
                qDes[idx] = lerp(qInit[idx], MID[j], r)
        # left-side stays at qInit throughout

    # Phase 3: swing right legs, soft-hold everything else (≥400)
    else:
        sin_i += 1; t = DT*sin_i
        hip  =  HIP_AMP  * math.sin(ω*t)
        knee = -KNEE_AMP * math.sin(ω*t)

        # start from captured pose
        qDes[:] = qInit.copy()
        # overwrite swing joints
        qDes[d['FR_1']] = MID[1] + hip
        qDes[d['FR_2']] = MID[2] + knee
        qDes[d['RR_1']] = MID[1] + hip
        qDes[d['RR_2']] = MID[2] + knee
        # keep FR_0 & RR_0 at MID[0] (roll neutral)
        qDes[d['FR_0']] = MID[0]
        qDes[d['RR_0']] = MID[0]

    # ── fill LowCmd with two gain levels ───────────────
    for name,idx in d.items():
        moving = name in SWING
        kp = SWING_KP if moving else SOFT_KP
        kd = SWING_KD if moving else SOFT_KD
        cmd.motorCmd[idx].q   = qDes[idx]
        cmd.motorCmd[idx].dq  = 0
        cmd.motorCmd[idx].Kp  = kp
        cmd.motorCmd[idx].Kd  = kd
        cmd.motorCmd[idx].tau = 0

    # light inward bias on hip-rolls (keeps knees clear)
    cmd.motorCmd[d['FR_0']].tau = -0.4
    cmd.motorCmd[d['RR_0']].tau = -0.4

    if motion > 10:  safe.PowerProtect(cmd, state, 1)
    udp.SetSend(cmd); udp.Send()
