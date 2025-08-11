#!/usr/bin/python
#
# “Push-leg” demo — swing FR leg front ↔ back in sagittal plane.
# Author: <you>

import sys, time, math, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk                     # ← needs the SDK in that folder

# ---------------------- tunables ----------------------
PITCH_AMP = 0.40          # rad, ± about sin_mid_q[1]
KNEE_AMP  = 0.25          # rad, opposite sign of hip-pitch
freq_Hz   = 1.0           # push frequency
KP        = [20, 20, 20]  # PD gains during motion
KD        = [ 2,  2,  2]
# ------------------------------------------------------

d = {'FR_0':0,'FR_1':1,'FR_2':2,'FL_0':3,'FL_1':4,'FL_2':5,
     'RR_0':6,'RR_1':7,'RR_2':8,'RL_0':9,'RL_1':10,'RL_2':11}

PosStopF = math.pow(10,9);  VelStopF = 16000.0
LOWLEVEL = 0xff
DT       = 0.002

# “neutral mid-pose” that looks like a comfortable stance before we start swinging
sin_mid_q = [0.00, 1.2, -2.0]                     # [roll, pitch, knee] target in rad

def lerp(a, b, r):                                # simple linear blend 0-1
    r = np.fmin(np.fmax(r, 0.0), 1.0)
    return a*(1-r) + b*r

if __name__ == '__main__':
    udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)

    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    qInit = [0.0, 0.0, 0.0];  qDes = [0.0, 0.0, 0.0]
    motiontime = 0;  ramp_i = 0;  sin_i = 0
    freq_rad = 2*math.pi*freq_Hz

    while True:
        time.sleep(DT)
        motiontime += 1
        udp.Recv();  udp.GetRecv(state)

        # ---------- Phase 1: sample current pose (first 10 cycles) ----------
        if motiontime < 10:
            qInit[0] = state.motorState[d['FR_0']].q
            qInit[1] = state.motorState[d['FR_1']].q
            qInit[2] = state.motorState[d['FR_2']].q

        # ---------- Phase 2: glide to mid-pose over 0.39 s ----------
        elif motiontime < 400:
            ramp_i += 1
            rate = ramp_i / 200.0                  # counts 0 → 1 over 200 loops
            qDes[0] = lerp(qInit[0], sin_mid_q[0], rate)
            qDes[1] = lerp(qInit[1], sin_mid_q[1], rate)
            qDes[2] = lerp(qInit[2], sin_mid_q[2], rate)

        # ---------- Phase 3: swing front ↔ back forever ----------
        else:
            sin_i += 1
            t = DT * sin_i
            hip_pitch =  PITCH_AMP * math.sin(freq_rad*t)
            knee      = -KNEE_AMP  * math.sin(freq_rad*t)
            qDes[0] = sin_mid_q[0]                # hip-roll locked
            qDes[1] = sin_mid_q[1] + hip_pitch
            qDes[2] = sin_mid_q[2] + knee

        # ------------- fill LowCmd for the three FR joints -------------
        for j, idx in enumerate(('FR_0','FR_1','FR_2')):
            i = d[idx]
            cmd.motorCmd[i].q   = qDes[j]
            cmd.motorCmd[i].dq  = 0.0
            cmd.motorCmd[i].Kp  = KP[j]
            cmd.motorCmd[i].Kd  = KD[j]
            cmd.motorCmd[i].tau = 0.0
        cmd.motorCmd[d['FR_0']].tau = -0.65       # small assist to hip-roll

        # optional: stop other nine joints from drifting
        for i in range(12):
            if i in (d['FR_0'],d['FR_1'],d['FR_2']): continue
            cmd.motorCmd[i].q   = PosStopF
            cmd.motorCmd[i].dq  = VelStopF
            cmd.motorCmd[i].Kp  = 0
            cmd.motorCmd[i].Kd  = 0
            cmd.motorCmd[i].tau = 0

        if motiontime > 10:
            safe.PowerProtect(cmd, state, 1)

        udp.SetSend(cmd);  udp.Send()

