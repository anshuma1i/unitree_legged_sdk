#!/usr/bin/python
# push_board_side.py (smooth + stiff hold + cruise to lie_base)
# Skateboard push test: side-leg push (lying posture, asymmetric constrained)

import sys, time, numpy as np, os
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ========================
# USER PARAMETERS
# ========================
AMP          = 0.25   # radians, swing amplitude
SWING_TIME   = 3.0    # seconds per swing cycle
SWING_COUNT  = 3      # number of push cycles
RAMP_TIME    = 5.0    # seconds for smooth ramp
KP_HOLD      = 6.0    # joint stiffness
KD_HOLD      = 0.3    # damping
WAIT_BASE    = 10     # seconds to place robot on board
WAIT_SIDE    = 10     # seconds before swinging

# ========================
# NETWORK CONFIG
# ========================
LOWLEVEL   = 0xff
LOCAL_PORT = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
DT = 0.002

# Right-side legs indices
FR = 2  # front-right hip
HR = 8  # hind-right hip

def set_joint(cmd, idx, q, kp, kd, tau=0.0):
    m = cmd.motorCmd[idx]
    m.q, m.dq, m.tau = float(q), 0.0, float(tau)
    m.Kp, m.Kd = float(kp), float(kd)

def ramp_to_pose(udp, safe, cmd, state, target_pose, ramp_time):
    """Smoothly ramp into target pose from current joint angles"""
    start_pose = np.array([state.motorState[i].q for i in range(12)])
    start_time = time.time()
    while True:
        loop = time.time()
        udp.Recv(); udp.GetRecv(state)
        t = time.time() - start_time
        alpha = min(1.0, t / ramp_time)
        interp_pose = (1 - alpha) * start_pose + alpha * target_pose
        for i in range(12):
            set_joint(cmd, i, interp_pose[i], KP_HOLD, KD_HOLD)
        safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd); udp.Send()
        if alpha >= 1.0:
            break
        time.sleep(max(0.0, DT - (time.time() - loop)))

def hold_pose(udp, safe, cmd, state, target_pose, hold_time):
    """Hold a given pose stiffly for hold_time seconds"""
    start_time = time.time()
    while True:
        loop = time.time()
        udp.Recv(); udp.GetRecv(state)
        for i in range(12):
            set_joint(cmd, i, target_pose[i], KP_HOLD, KD_HOLD)
        safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd); udp.Send()
        if time.time() - start_time >= hold_time:
            break
        time.sleep(max(0.0, DT - (time.time() - loop)))

def main():
    if not os.path.exists("lie_base.npy") or not os.path.exists("lie_skate_right.npy"):
        print("✗ Required pose files not found. Run record scripts first.")
        return

    base_pose = np.load("lie_base.npy")
    side_pose = np.load("lie_skate_right.npy")

    udp   = sdk.UDP(LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    safe  = sdk.Safety(sdk.LeggedType.Go1)
    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    # Warm-up
    for _ in range(10):
        udp.Recv(); udp.GetRecv(state); time.sleep(0.005)

    try:
        # Step 1: Move into base pose smoothly
        print("► Moving to lie_base pose (smooth)...")
        ramp_to_pose(udp, safe, cmd, state, base_pose, RAMP_TIME)
        print(f"✓ Holding lie_base. You have {WAIT_BASE}s to place robot on board.")
        hold_pose(udp, safe, cmd, state, base_pose, WAIT_BASE)

        # Step 2: Move into skate-right pose smoothly
        print("► Moving to lie_skate_right pose (smooth)...")
        ramp_to_pose(udp, safe, cmd, state, side_pose, RAMP_TIME)
        print(f"✓ Holding lie_skate_right. Starting swing in {WAIT_SIDE}s...")
        hold_pose(udp, safe, cmd, state, side_pose, WAIT_SIDE)

        # Step 3: Swing phase
        start_pose = side_pose.copy()
        q_base_FR = start_pose[FR]
        q_base_HR = start_pose[HR]

        swing_start = time.time()
        total_time = SWING_TIME * SWING_COUNT
        print("► Swinging right legs with constrained asymmetric motion...")

        while True:
            loop = time.time()
            udp.Recv(); udp.GetRecv(state)

            t = time.time() - swing_start
            if t > total_time:
                break

            interp_pose = start_pose.copy()
            phase = (t % SWING_TIME) / SWING_TIME

            # --- Front Right ---
            if phase < 0.7:   # push phase
                offset = -AMP * (phase / 0.7)
                q_fr = q_base_FR + offset
                q_fr = min(q_fr, q_base_FR)  # don’t go beyond contact
            else:              # recovery
                offset = +AMP * ((phase-0.7)/0.3)
                q_fr = q_base_FR + offset

            # --- Hind Right ---
            if phase < 0.7:
                offset = -AMP * (phase / 0.7)
                q_hr = q_base_HR + offset
                q_hr = min(q_hr, q_base_HR)
            else:
                offset = +AMP * ((phase-0.7)/0.3)
                q_hr = q_base_HR + offset

            interp_pose[FR] = q_fr
            interp_pose[HR] = q_hr

            for i in range(12):
                set_joint(cmd, i, interp_pose[i], KP_HOLD, KD_HOLD)

            safe.PowerProtect(cmd, state, 1)
            udp.SetSend(cmd); udp.Send()
            time.sleep(max(0.0, DT - (time.time() - loop)))

        # Step 4: End with all 4 legs lifted (cruise = lie_base)
        print("► Transitioning into cruising pose (all legs lifted)...")
        ramp_to_pose(udp, safe, cmd, state, base_pose, 2.0)
        print("✓ Push done. Holding cruising pose indefinitely.")

        while True:
            loop = time.time()
            udp.Recv(); udp.GetRecv(state)
            for i in range(12):
                set_joint(cmd, i, base_pose[i], KP_HOLD, KD_HOLD)
            safe.PowerProtect(cmd, state, 1)
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
