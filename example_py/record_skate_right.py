#!/usr/bin/python
# record_skate_right.py
# Record the right-side push-ready pose as lie_skate_right.npy

import sys, time, numpy as np, os
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

LOWLEVEL   = 0xff
LOCAL_PORT = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
DT = 0.002

KP_LIMP, KD_LIMP = 0.0, 0.08
KP_SOFT_HOLD, KD_SOFT_HOLD = 2.0, 0.20

# mapping
d = {f'{leg}_{j}': i for i,(leg,j) in enumerate([(L,k) for L in ('FR','FL','RR','RL') for k in range(3)])}
LEFT_LEGS  = ('FL','RL')
RIGHT_LEGS = ('FR','RR')

def set_joint(cmd, idx, q, kp, kd, tau=0.0):
    m = cmd.motorCmd[idx]
    m.q, m.dq, m.tau = float(q), 0.0, float(tau)
    m.Kp, m.Kd = float(kp), float(kd)

def main():
    if not os.path.exists("lie_base.npy"):
        print("✗ lie_base.npy not found. Run record_base.py first.")
        return
    lie_base = np.load("lie_base.npy")

    udp   = sdk.UDP(LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    safe  = sdk.Safety(sdk.LeggedType.Go1)
    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    for _ in range(10):
        udp.Recv(); udp.GetRecv(state); time.sleep(0.005)

    print("► Step 2: Keep LEFT legs at lie_base; move RIGHT legs to push-ready. You have ~30s...")
    t0 = time.time()
    while (time.time() - t0) < 30.0:
        loop = time.time()
        udp.Recv(); udp.GetRecv(state)

        # left hold
        for L in LEFT_LEGS:
            for j in range(3):
                idx = d[f'{L}_{j}']
                set_joint(cmd, idx, lie_base[idx], KP_SOFT_HOLD, KD_SOFT_HOLD)

        # right limp
        for R in RIGHT_LEGS:
            for j in range(3):
                idx = d[f'{R}_{j}']
                q_now = state.motorState[idx].q
                set_joint(cmd, idx, q_now, KP_LIMP, KD_LIMP)

        safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd); udp.Send()
        time.sleep(max(0.0, DT - (time.time() - loop)))

    udp.Recv(); udp.GetRecv(state)
    lie_skate_right = np.array([state.motorState[i].q for i in range(12)], dtype=float)
    np.save("lie_skate_right.npy", lie_skate_right)
    print("✓ Saved lie_skate_right.npy")

    # relax
    for i in range(12):
        set_joint(cmd, i, state.motorState[i].q, 0.0, 0.0)
    udp.SetSend(cmd); udp.Send()
    print("Done.")

if __name__ == "__main__":
    main()
