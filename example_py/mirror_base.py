#!/usr/bin/python
# mirror_base.py
# Smoothly ramp into lie_base.npy pose and hold indefinitely

import sys, time, numpy as np, os
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

LOWLEVEL   = 0xff
LOCAL_PORT = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
DT = 0.002

KP_HOLD, KD_HOLD = 6.0, 0.3   # holding gains
RAMP_TIME = 3.0               # seconds to ramp into pose

def set_joint(cmd, idx, q, kp, kd, tau=0.0):
    m = cmd.motorCmd[idx]
    m.q, m.dq, m.tau = float(q), 0.0, float(tau)
    m.Kp, m.Kd = float(kp), float(kd)

def main():
    if not os.path.exists("lie_base.npy"):
        print("✗ lie_base.npy not found. Run record_base.py first.")
        return
    target_pose = np.load("lie_base.npy")

    udp   = sdk.UDP(LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    safe  = sdk.Safety(sdk.LeggedType.Go1)
    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    # Warm-up
    for _ in range(10):
        udp.Recv(); udp.GetRecv(state); time.sleep(0.005)

    # Capture start pose (current joint states)
    start_pose = np.array([state.motorState[i].q for i in range(12)])
    start_time = time.time()
    print("► Ramping smoothly into lie_base pose...")

    try:
        while True:
            loop = time.time()
            udp.Recv(); udp.GetRecv(state)

            t = time.time() - start_time
            alpha = min(1.0, t / RAMP_TIME)
            interp_pose = (1 - alpha) * start_pose + alpha * target_pose

            for i in range(12):
                set_joint(cmd, i, interp_pose[i], KP_HOLD, KD_HOLD)

            # less strict safety during ramp
            safe.PowerProtect(cmd, state, 0)
            udp.SetSend(cmd); udp.Send()
            time.sleep(max(0.0, DT - (time.time() - loop)))

    except KeyboardInterrupt:
        print("\nExiting, releasing joints...")
        for i in range(12):
            set_joint(cmd, i, state.motorState[i].q, 0.0, 0.0)
        udp.SetSend(cmd); udp.Send()
        print("Done.")

if __name__ == "__main__":
    main()
