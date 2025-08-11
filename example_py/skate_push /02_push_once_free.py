#!/usr/bin/python3
"""
01_push_once_free.py
Rear-leg push *without freezing* the front legs.
Front joints get a light PD (Kp=5) to stay near current angles but
can yield if the board moves.
"""

import sys, time, math, pathlib, numpy as np
SDK_PY = pathlib.Path(__file__).resolve().parents[2] / "lib/python/amd64"
sys.path.append(str(SDK_PY))
import robot_interface as sdk

LOWLEVEL = 0xff
DT       = 0.002
CYCLE    = 1.5
STEPS    = int(CYCLE/DT)

# joint index groups
front = [0,1,2, 3,4,5]
rear  = [6,7,8, 9,10,11]

# ---- trajectory for rear legs --------------------------------
base = np.array([-0.25, 1.05, -1.80,   # RR hip thigh calf
                 -0.25, 1.05, -1.80])  # RL
FREQ = 1.0/CYCLE
AMP_HIP, AMP_THIGH, AMP_CALF = -0.30, 0.45, -0.65
PHASE = math.pi

t = np.arange(STEPS)*DT
traj = np.zeros((STEPS,6))
traj[:,0] = base[0] + AMP_HIP *np.sin(2*math.pi*FREQ*t)
traj[:,1] = base[1] + AMP_THIGH*np.sin(2*math.pi*FREQ*t)
traj[:,2] = base[2] + AMP_CALF*np.sin(2*math.pi*FREQ*t)
traj[:,3] = base[3] + AMP_HIP *np.sin(2*math.pi*FREQ*t + PHASE)
traj[:,4] = base[4] + AMP_THIGH*np.sin(2*math.pi*FREQ*t + PHASE)
traj[:,5] = base[5] + AMP_CALF*np.sin(2*math.pi*FREQ*t + PHASE)

# ---- UDP ------------------------------------------------------
udp  = sdk.UDP(LOWLEVEL,8080,"192.168.12.1",8007)
cmd  = sdk.LowCmd();  state = sdk.LowState()
udp.InitCmdData(cmd)

# capture front-leg starting pose once
print("▶  Capturing front-leg stance …")
for _ in range(10):
    udp.Recv(); udp.GetRecv(state); time.sleep(DT)
front_start = [state.motorState[i].q for i in front]
print("Front start:", [f"{a:+.2f}" for a in front_start])

print("▶  Push starts in 2 s …"); time.sleep(2)

for k in range(STEPS):
    udp.Recv(); udp.GetRecv(state)
    ramp = min(1.0, k/(1.0/DT))          # 1-s ramp

    # lightly hold front legs
    for idx, start_q in zip(front, front_start):
        mc = cmd.motorCmd[idx]
        mc.q   = start_q
        mc.dq  = 0.0
        mc.Kp  = 5.0                      # soft
        mc.Kd  = 1.0
        mc.tau = 0.0

    # rear-leg push
    for j, mid in enumerate(rear):
        target = base[j] + ramp*(traj[k,j]-base[j])
        mc = cmd.motorCmd[mid]
        mc.q, mc.dq = target, 0.0
        mc.Kp, mc.Kd = 12.0, 2.0
        mc.tau = 0.0

    # send & small IMU print every 50 cycles (~0.1 s)
    if k % 50 == 0:
        print(f"t={k*DT:4.2f}s pitch={state.imu.rpy[1]:+.2f} roll={state.imu.rpy[0]:+.2f}")
    udp.SetSend(cmd); udp.Send(); time.sleep(DT)

# limp all joints
for mc in cmd.motorCmd: mc.Kp = mc.Kd = mc.tau = 0
udp.SetSend(cmd); udp.Send()
print("Push finished; robot limp.")

