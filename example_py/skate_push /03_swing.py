#!/usr/bin/python
#
# 03_swing_side.py  – front-right *and* rear-right legs push-style sweep
# based on example_position → 01_swing, amplitudes stay joint-safe. 

import sys, time, math, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ─────────────── user-tunable parameters ───────────────
HIP_AMP    = 0.55          # rad, ± around neutral hip-pitch
KNEE_GAIN  = 0.65          # knee_amp = KNEE_GAIN * HIP_AMP  (opposite phase)
FREQ_HZ    = 1.2           # sweep frequency
KP         = [25, 25, 25]  # PD gains for each joint we drive
KD         = [ 3,  3,  3]
# ────────────────────────────────────────────────────────

d = {'FR_0':0,'FR_1':1,'FR_2':2,'FL_0':3,'FL_1':4,'FL_2':5,
     'RR_0':6,'RR_1':7,'RR_2':8,'RL_0':9,'RL_1':10,'RL_2':11}

PosStopF = 1e9;   VelStopF = 16000.0
LOWLEVEL = 0xff;  DT = 0.002
# neutral “ready” pose copied from example_position / 01_swing
sin_mid_q = [0.0, 1.2, -2.0]                      # [roll, pitch, knee]

def lerp(a,b,r): r=max(0,min(1,r)); return a*(1-r)+b*r

# which joints we actively command each loop
ACTIVE = ('FR_0','FR_1','FR_2','RR_0','RR_1','RR_2')

udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd, state = sdk.LowCmd(), sdk.LowState()
udp.InitCmdData(cmd)

qInit = np.zeros(12);  qDes = np.zeros(12)
motion = ramp_i = sin_i = 0
omega = 2*math.pi*FREQ_HZ
KNEE_AMP = KNEE_GAIN * HIP_AMP

while True:
    time.sleep(DT); motion += 1
    udp.Recv(); udp.GetRecv(state)

    # Phase 1 – sample stand pose
    if motion < 10:
        for k,i in d.items(): qInit[i] = state.motorState[i].q

    # Phase 2 – glide to neutral on both right legs
    elif motion < 400:
        ramp_i += 1; r = ramp_i/200.0
        for leg in ('FR','RR'):
            for j,off in zip((0,1,2),sin_mid_q):
                idx = d[f'{leg}_{j}']
                qDes[idx] = lerp(qInit[idx], off, r)

    # Phase 3 – continuous sinusoid on right legs
    else:
        sin_i += 1; t = DT*sin_i
        hip  =  HIP_AMP  * math.sin(omega*t)
        knee = -KNEE_AMP * math.sin(omega*t)
        for leg in ('FR','RR'):
            qDes[d[f'{leg}_0']] = sin_mid_q[0]       # roll locked
            qDes[d[f'{leg}_1']] = sin_mid_q[1] + hip
            qDes[d[f'{leg}_2']] = sin_mid_q[2] + knee

    # ───── fill LowCmd ─────
    for k in ACTIVE:
        i = d[k]; j = i%3            # map to 0,1,2 for KP/KD lists
        cmd.motorCmd[i].q  = qDes[i]
        cmd.motorCmd[i].dq = 0
        cmd.motorCmd[i].Kp = KP[j]
        cmd.motorCmd[i].Kd = KD[j]
        cmd.motorCmd[i].tau = 0
    # bias hip-rolls slightly inward for both right legs
    cmd.motorCmd[d['FR_0']].tau = -0.65
    cmd.motorCmd[d['RR_0']].tau = -0.65

    # freeze untouched joints each loop
    for i in range(12):
        if list(d.keys())[i] in ACTIVE: continue
        m = cmd.motorCmd[i]; m.q=PosStopF; m.dq=VelStopF; m.Kp=m.Kd=m.tau=0

    if motion > 10: safe.PowerProtect(cmd,state,1)
    udp.SetSend(cmd); udp.Send()
