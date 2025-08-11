#!/usr/bin/python3
# ------------------------------------------------------------
# 01_push_once.py  --  Rear-leg kick with hip swing, ramped.
# Run while 00_lock_front.py is holding front legs.
# ------------------------------------------------------------
import sys, time, math, pathlib, numpy as np
SDK_PY = pathlib.Path(__file__).resolve().parents[2] / "lib/python/amd64"
sys.path.append(str(SDK_PY))
import robot_interface as sdk

LOWLEVEL = 0xff
DT       = 0.002
CYCLE    = 1.5                      # sec
STEPS    = int(CYCLE/DT)

rear = [6,7,8, 9,10,11]             # RR hip,thigh,calf, RL hip,thigh,calf
# base crouch pose (hip, thigh, calf) ×2
base = np.array([-0.25, 1.05, -1.8,
                 -0.25, 1.05, -1.8])

# sine parameters
FREQ  = 1.0/CYCLE
AMP_HIP, AMP_THIGH, AMP_CALF = -0.35, 0.55, -0.75
PHASE = math.pi               # 180° offset left vs right

# build trajectory table
t = np.arange(STEPS)*DT
traj = np.zeros((STEPS,6))
traj[:,0] = base[0] + AMP_HIP *np.sin(2*math.pi*FREQ*t)
traj[:,1] = base[1] + AMP_THIGH*np.sin(2*math.pi*FREQ*t)
traj[:,2] = base[2] + AMP_CALF*np.sin(2*math.pi*FREQ*t)
traj[:,3] = base[3] + AMP_HIP *np.sin(2*math.pi*FREQ*t + PHASE)
traj[:,4] = base[4] + AMP_THIGH*np.sin(2*math.pi*FREQ*t + PHASE)
traj[:,5] = base[5] + AMP_CALF*np.sin(2*math.pi*FREQ*t + PHASE)

# UDP init
udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
cmd  = sdk.LowCmd(); state = sdk.LowState()
udp.InitCmdData(cmd)

print("▶  Push begins in 1 s…"); time.sleep(1)

for k in range(STEPS):
    udp.Recv(); udp.GetRecv(state)
    ramp = min(1.0, k/(0.5/DT))                # 0→1 over 0.5 s

    for j, mid in enumerate(rear):
        target = base[j] + ramp*(traj[k,j]-base[j])
        mc = cmd.motorCmd[mid]
        mc.q, mc.dq  = target, 0.0
        mc.Kp, mc.Kd = 14.0, 2.5               # moderate stiffness
        mc.tau       = 0.0

    udp.SetSend(cmd); udp.Send()
    time.sleep(DT)

print("Push done.  Limping joints.")
for mc in cmd.motorCmd: mc.Kp = mc.Kd = mc.tau = 0
udp.SetSend(cmd); udp.Send()

