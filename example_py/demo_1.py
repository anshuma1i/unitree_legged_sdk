#!/usr/bin/python
# push_board_hind.py (with "Catch and Hold" Transition)
# Features a seamless transition from the final push to the cruise pose to eliminate drag.

import sys, time, numpy as np, os
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ========================
# USER PARAMETERS (Your preferred tune)
# ========================
# --- Force & Motion Tuning ---
TAU_BOOST     = 10.0 # Feed-forward torque in Nm applied to the hips during the push.
Z_OFFSET_KNEE = 0.25 # MODIFIED: Slightly reduced downward pressure (was 0.25).
AMP_PUSH      = 0.55 # A long backward stroke.
AMP_REC       = 0.45 # Matching recovery stroke.
PUSH_RATIO    = 0.75 # 75% of the cycle is the power stroke.
SWING_TIME    = 0.75 # MODIFIED: Very fast swing cycle time (was 2.5s).
SWING_COUNT   = 20   # Perform 20 push cycles.

# --- Gains ---
KP_PUSH_HIP   = 25.0  # High stiffness to maintain path during high torque application.
KNEE_GAIN     = 1.2   # Strong knee synergy.
KP_HOLD_OTHER = 8.0   # Stiffness for non-moving joints.
KD_HOLD_OTHER = 0.7   # MODIFIED: Increased damping for stability with very fast motion (was 0.6).
KP_REC_HIP    = 7.0   # MODIFIED: Increased recovery stiffness for the very fast return stroke (was 6.0).
KNEE_TAU_S    = 0.06  # Smoothing filter.

# --- Sequence Timing ---
RAMP_TIME_DEFAULT = 5.0 # Default ramp time.
RAMP_TIME_GROUND  = 3.0 # Slow ramp for applying pressure.
WAIT_BASE     = 5 # sec to place robot on board.
WAIT_HIND     = 5  # sec before applying pressure.

# ========================
# NETWORK CONFIG
# ========================
LOWLEVEL   = 0xff
LOCAL_PORT = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
DT = 0.002

# ---- Joint mapping ----
order = [(L,k) for L in ('FR','FL','RR','RL') for k in range(3)]
d = {f'{leg}_{j}': i for i,(leg,j) in enumerate(order)}
HL_HIP, HR_HIP = d['RL_1'], d['RR_1']
HL_KNEE, HR_KNEE = d['RL_2'], d['RR_2']

# ----------------- helpers -----------------
def s_curve(a: float) -> float:
    a = max(0.0, min(1.0, a))
    return 10*a**3 - 15*a**4 + 6*a**5

def set_joint(cmd, idx, q, kp, kd, tau=0.0):
    m = cmd.motorCmd[idx]
    m.q, m.dq, m.tau = float(q), 0.0, float(tau)
    m.Kp, m.Kd = float(kp), float(kd)

def ramp_to_pose(udp, safe, cmd, state, target_pose, ramp_time, kp=KP_HOLD_OTHER, kd=KD_HOLD_OTHER):
    start_pose = np.array([state.motorState[i].q for i in range(12)], dtype=float)
    t0 = time.time()
    while True:
        loop = time.time()
        udp.Recv(); udp.GetRecv(state)
        a = s_curve((time.time() - t0) / ramp_time)
        interp = (1 - a) * start_pose + a * target_pose
        for i in range(12): set_joint(cmd, i, interp[i], kp, kd)
        safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd); udp.Send()
        if a >= 1.0: break
        time.sleep(max(0.0, DT - (time.time() - loop)))

def hold_pose(udp, safe, cmd, state, target_pose, hold_time, kp=KP_HOLD_OTHER, kd=KD_HOLD_OTHER):
    t0 = time.time()
    while True:
        loop = time.time()
        udp.Recv(); udp.GetRecv(state)
        for i in range(12): set_joint(cmd, i, target_pose[i], kp, kd)
        safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd); udp.Send()
        if time.time() - t0 >= hold_time: break
        time.sleep(max(0.0, DT - (time.time() - loop)))

# ----------------- main -----------------
def main():
    if not os.path.exists("lie_base.npy") or not os.path.exists("lie_skate_hind.npy"):
        print("✗ Required pose files not found. Run record scripts first.")
        return

    base_pose = np.load("lie_base.npy").astype(float)
    hind_pose_recorded = np.load("lie_skate_hind.npy").astype(float)

    pressurized_pose = hind_pose_recorded.copy()
    pressurized_pose[HL_KNEE] += Z_OFFSET_KNEE
    pressurized_pose[HR_KNEE] += Z_OFFSET_KNEE
    
    udp   = sdk.UDP(LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    safe  = sdk.Safety(sdk.LeggedType.Go1)
    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    for _ in range(10): udp.Recv(); udp.GetRecv(state); time.sleep(0.005)

    try:
        # 1) Base pose
        print("► Moving to lie_base pose…")
        ramp_to_pose(udp, safe, cmd, state, base_pose, RAMP_TIME_DEFAULT)
        print(f"✓ Holding lie_base for {WAIT_BASE}s.")
        hold_pose(udp, safe, cmd, state, base_pose, WAIT_BASE)

        # 2) Grounding Sequence
        print("► Moving to recorded hind_pose to make contact…")
        ramp_to_pose(udp, safe, cmd, state, hind_pose_recorded, RAMP_TIME_DEFAULT)
        print(f"✓ Holding recorded pose for {WAIT_HIND}s.")
        hold_pose(udp, safe, cmd, state, hind_pose_recorded, WAIT_HIND)

        print(f"► Applying downward pressure over {RAMP_TIME_GROUND}s…")
        ramp_to_pose(udp, safe, cmd, state, pressurized_pose, RAMP_TIME_GROUND)
        print("✓ Pressure applied. Starting push cycles.")
        
        # 3) Push cycles with seamless transition
        q_base_HL_hip, q_base_HR_hip = pressurized_pose[HL_HIP], pressurized_pose[HR_HIP]
        q_base_HL_knee, q_base_HR_knee = pressurized_pose[HL_KNEE], pressurized_pose[HR_KNEE]
        knee_hl_off, knee_hr_off = 0.0, 0.0
        alpha_lpf = DT / max(1e-3, KNEE_TAU_S)

        # MODIFIED: State variables for the new transition logic
        transition_started = False
        transition_start_pose = np.zeros(12)
        transition_start_time = 0.0
        final_recovery_start_time = SWING_TIME * (SWING_COUNT - 1) + SWING_TIME * PUSH_RATIO
        total_time = SWING_TIME * SWING_COUNT

        t0 = time.time()
        print(f"► Swinging with TAU_BOOST = {TAU_BOOST} Nm…")

        while True:
            loop = time.time()
            udp.Recv(); udp.GetRecv(state)
            t = time.time() - t0

            if t > total_time:
                break

            # MODIFIED: Logic to detect and handle the final transition
            if not transition_started and t >= final_recovery_start_time:
                print("► Catching final recovery stroke for seamless transition…")
                transition_started = True
                transition_start_time = t
                # Capture the current full-body pose to serve as the start of our smooth ramp
                transition_start_pose = np.array([state.motorState[i].q for i in range(12)])

            if transition_started:
                # We are in the final recovery phase; override normal logic to ramp smoothly to the cruise pose.
                time_in_transition = t - transition_start_time
                transition_duration = SWING_TIME * (1.0 - PUSH_RATIO)
                alpha = s_curve(time_in_transition / max(1e-3, transition_duration))

                # Interpolate ALL joints from where they were to the final base_pose.
                final_pose = (1 - alpha) * transition_start_pose + alpha * base_pose
                for i in range(12):
                    set_joint(cmd, i, final_pose[i], KP_REC_HIP, KD_HOLD_OTHER) # Use gentle gains for the transition
            else:
                # This is the normal push/recovery logic for all but the final recovery stroke.
                pose = pressurized_pose.copy()
                phase = (t % SWING_TIME) / SWING_TIME
                hip_tau = 0.0

                if phase < PUSH_RATIO:
                    ap = s_curve(phase / PUSH_RATIO)
                    hip_off = AMP_PUSH * ap
                    kp_hip  = KP_PUSH_HIP
                    hip_tau = TAU_BOOST
                else:
                    ar = s_curve((phase - PUSH_RATIO) / (1.0 - PUSH_RATIO))
                    hip_off = AMP_PUSH - (AMP_PUSH + AMP_REC) * ar
                    kp_hip  = KP_REC_HIP

                pose[HL_HIP] = max(q_base_HL_hip + hip_off, q_base_HL_hip)
                pose[HR_HIP] = max(q_base_HR_hip + hip_off, q_base_HR_hip)
                knee_target = -KNEE_GAIN * hip_off
                knee_hl_off = (1 - alpha_lpf) * knee_hl_off + alpha_lpf * knee_target
                knee_hr_off = (1 - alpha_lpf) * knee_hr_off + alpha_lpf * knee_target
                pose[HL_KNEE] = q_base_HL_knee + knee_hl_off
                pose[HR_KNEE] = q_base_HR_knee + knee_hr_off

                for i in range(12):
                    if i == HL_HIP or i == HR_HIP:
                        set_joint(cmd, i, pose[i], kp_hip, KD_HOLD_OTHER, tau=hip_tau)
                    else:
                        set_joint(cmd, i, pose[i], KP_HOLD_OTHER, KD_HOLD_OTHER)

            safe.PowerProtect(cmd, state, 1)
            udp.SetSend(cmd); udp.Send()
            time.sleep(max(0.0, DT - (time.time() - loop)))

        # 4) Cruise back
        # The jarring ramp is no longer needed as the transition is now handled inside the loop.
        print("✓ Push cycles and seamless transition complete. Holding cruising pose.")
        while True:
            loop = time.time()
            udp.Recv(); udp.GetRecv(state)
            # We just hold the final base_pose.
            for i in range(12): set_joint(cmd, i, base_pose[i], KP_HOLD_OTHER, KD_HOLD_OTHER)
            safe.PowerProtect(cmd, state, 1)
            udp.SetSend(cmd); udp.Send()
            time.sleep(max(0.0, DT - (time.time() - loop)))

    except KeyboardInterrupt:
        print("\nExiting, releasing joints…")
        for i in range(12): set_joint(cmd, i, state.motorState[i].q, 0.0, 0.0)
        udp.SetSend(cmd); udp.Send()
        print("Done.")

if __name__ == "__main__":
    main()