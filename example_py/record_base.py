#!/usr/bin/python
# record_base.py
# Record the lying-down base pose as lie_base.npy

import sys, time, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

LOWLEVEL   = 0xff
LOCAL_PORT = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
DT = 0.002

KP_LIMP, KD_LIMP = 0.0, 0.08

def set_joint(cmd, idx, q, kp, kd, tau=0.0):
    m = cmd.motorCmd[idx]
    m.q, m.dq, m.tau = float(q), 0.0, float(tau)
    m.Kp, m.Kd = float(kp), float(kd)

def main():
    udp   = sdk.UDP(LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    safe  = sdk.Safety(sdk.LeggedType.Go1)
    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    for _ in range(10):
        udp.Recv(); udp.GetRecv(state); time.sleep(0.005)

    print("► Place the robot lying flat. You have ~10s...")
    t0 = time.time()
    while (time.time() - t0) < 10.0:
        loop = time.time()
        udp.Recv(); udp.GetRecv(state)

        for i in range(12):
            q_now = state.motorState[i].q
            set_joint(cmd, i, q_now, KP_LIMP, KD_LIMP)

        safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd); udp.Send()
        time.sleep(max(0.0, DT - (time.time() - loop)))

    udp.Recv(); udp.GetRecv(state)
    lie_base = np.array([state.motorState[i].q for i in range(12)], dtype=float)
    np.save("lie_base.npy", lie_base)
    print("✓ Saved lie_base.npy")

    # relax
    for i in range(12):
        set_joint(cmd, i, state.motorState[i].q, 0.0, 0.0)
    udp.SetSend(cmd); udp.Send()
    print("Done.")

if __name__ == "__main__":
    main()
