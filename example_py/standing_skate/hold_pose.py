#!/usr/bin/python
# hold_pose_soft.py  – blend gently to stand_pose.npy while hanging
import sys, time, os, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

POSE = np.load('stand_pose.npy')          # 12-vector (saved on ground)

LOWLEVEL = 0xff;  DT=0.002
udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd, state = sdk.LowCmd(), sdk.LowState();  udp.InitCmdData(cmd)

# ---- blend parameters ----
RAMP_SEC     = 3.0          # seconds to reach full pose & gains
KP_START, KD_START = 4.0, 0.3
KP_END,   KD_END   = 12.0, 1.0
# ---------------------------

t0 = time.time()
print("► Low-level stream active. Robot should be limp & hanging.")

while True:
    loop = time.time()
    udp.Recv(); udp.GetRecv(state)

    t = loop - t0
    alpha = min(t / RAMP_SEC, 1.0)        # 0 → 1 over RAMP_SEC
    kp = KP_START + alpha*(KP_END-KP_START)
    kd = KD_START + alpha*(KD_END-KD_START)

    for i in range(12):
        cur_q = state.motorState[i].q
        target_q = cur_q*(1-alpha) + POSE[i]*alpha   # position ramp
        m = cmd.motorCmd[i]
        m.q, m.dq = target_q, 0.0
        m.Kp, m.Kd = kp, kd
        m.tau = 0.0

    # mild inward roll bias
    cmd.motorCmd[0].tau = -0.3   # FR_0
    cmd.motorCmd[6].tau = -0.3   # RR_0

    # enable safety after 1 s so current has settled
    if t > 1.0:
        safe.PowerProtect(cmd, state, 1)

    udp.SetSend(cmd); udp.Send()
    # keep cycle 2 ms
    sleep = DT - (time.time() - loop)
    if sleep > 0: time.sleep(sleep)
