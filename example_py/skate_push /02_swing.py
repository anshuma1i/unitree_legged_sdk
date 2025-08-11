#!/usr/bin/python
#
# “Push-leg v2” — deeper front-right sweep (still joint-safe)
import sys, time, math, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ---------- tunables you’re likely to tweak ----------
HIP_AMP   = 0.55          # rad, ± about sin_mid_q[1]  (↑ for deeper stroke)
KNEE_GAIN = 0.65          # knee_amp = KNEE_GAIN * HIP_AMP   (opposite phase)
FREQ_HZ   = 1.2           # leg cycles per second
KP        = [25, 25, 25]  # slightly stiffer than v1
KD        = [ 3,  3,  3]
# ------------------------------------------------------

d = {'FR_0':0,'FR_1':1,'FR_2':2,'FL_0':3,'FL_1':4,'FL_2':5,
     'RR_0':6,'RR_1':7,'RR_2':8,'RL_0':9,'RL_1':10,'RL_2':11}
PosStopF = 1e9;  VelStopF = 16000.0
LOWLEVEL = 0xff;  DT = 0.002
sin_mid_q = [0.0, 1.2, -2.0]                      # comfortable neutral

def lerp(a,b,r): return a*(1-r)+b*r if 0<=r<=1 else (b if r>1 else a)

udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd, state = sdk.LowCmd(), sdk.LowState()
udp.InitCmdData(cmd)

qInit=[0,0,0]; qDes=[0,0,0]; motion=0; ramp_i=0; sin_i=0
omega = 2*math.pi*FREQ_HZ
KNEE_AMP = KNEE_GAIN * HIP_AMP                    # derived once

while True:
    time.sleep(DT);  motion += 1
    udp.Recv(); udp.GetRecv(state)

    # ---- Phase 1: sample stand pose ----
    if motion < 10:
        for j,k in enumerate(('FR_0','FR_1','FR_2')):
            qInit[j] = state.motorState[d[k]].q

    # ---- Phase 2: glide to mid-pose ----
    elif motion < 400:
        ramp_i += 1; r = ramp_i/200.0
        for j in range(3):
            qDes[j] = lerp(qInit[j], sin_mid_q[j], r)

    # ---- Phase 3: deeper sine sweep ----
    else:
        sin_i += 1; t = DT*sin_i
        hip   =  HIP_AMP  * math.sin(omega*t)
        knee  = -KNEE_AMP * math.sin(omega*t)
        qDes  = [sin_mid_q[0], sin_mid_q[1]+hip, sin_mid_q[2]+knee]

    # ---- fill LowCmd for the three FR joints ----
    for j,idx in enumerate(('FR_0','FR_1','FR_2')):
        i=d[idx]; cmd.motorCmd[i].q=qDes[j]; cmd.motorCmd[i].dq=0
        cmd.motorCmd[i].Kp=KP[j]; cmd.motorCmd[i].Kd=KD[j]; cmd.motorCmd[i].tau=0
    cmd.motorCmd[d['FR_0']].tau = -0.65            # bias to hold roll

    # stop other 9 joints drifting
    for i in range(12):
        if i in (d['FR_0'],d['FR_1'],d['FR_2']): continue
        m=cmd.motorCmd[i]; m.q=PosStopF; m.dq=VelStopF; m.Kp=m.Kd=m.tau=0

    if motion > 10: safe.PowerProtect(cmd,state,1)
    udp.SetSend(cmd); udp.Send()
