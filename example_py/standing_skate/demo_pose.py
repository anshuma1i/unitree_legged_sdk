#!/usr/bin/python
# prep_and_record_pose_teach_hold_bias_60s.py
# Smooth ramp to stand → 60s teach & firm-hold (with outward hip bias + anti-drift) → record pose + IMU

import sys, time, os, math
import numpy as np

# --- Unitree SDK ---
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ========= Config =========
DT = 0.002
RAMP_TO_STAND_S      = 10.0
WAIT_BEFORE_RECORD_S = 60.0
STAND_FILE           = "stand_pose.npy"

# RIGHT (stance) legs: supportive
KP_R, KD_R = 16.0, 1.4

# LEFT legs while you MOVE them (hand-teach) — easy to move, but not floppy
KP_L_MOVE_HIPTHIGH, KD_L_MOVE_HIPTHIGH = 1.2, 0.12
KP_L_MOVE_KNEE,     KD_L_MOVE_KNEE     = 0.9, 0.10

# LEFT legs when latched (HOLD) — firm enough to stay put
KP_L_HOLD_HIPTHIGH, KD_L_HOLD_HIPTHIGH = 6.0, 0.35
KP_L_HOLD_KNEE,     KD_L_HOLD_KNEE     = 3.0, 0.20

# Teach/hold thresholds
VEL_THRESH = 0.10       # rad/s: below → considered "still"
DWELL_SEC  = 0.50       # how long it must be still to latch
ERR_DEADBAND = 0.02     # rad: allowed error before anti-drift kicks in
KP_BOOST   = 3.0        # extra Kp during anti-drift burst
BOOST_TIME = 0.4        # seconds of temporary boost

# Outward hip torque bias to fight inward sag on LEFT hips (abduction)
# If the hip still drifts INWARD, increase HIP_BIAS_TAU (e.g., 0.6 → 0.9),
# or flip HIP_BIAS_SIGN = -1 if the sign is wrong for your robot.
HIP_BIAS_TAU  = 0.6
HIP_BIAS_SIGN = +1  # set to -1 if it pushes the wrong way

# Networking
LOWLEVEL  = 0xff
LOCAL_PORT = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
# =========================

# Joint map: (FR, FL, RR, RL) × (0..2)  ->  0..11
d = {f'{leg}_{j}': i for i,(leg,j) in enumerate([(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}
LEFT_LEGS  = ('FL','RL')
RIGHT_LEGS = ('FR','RR')

def s_curve(a):  # min-jerk 0..1
    return 3*a*a - 2*a*a*a

def set_joint(cmd, idx, q, kp, kd, tau=0.0):
    m = cmd.motorCmd[idx]
    m.q, m.dq, m.tau = float(q), 0.0, float(tau)
    m.Kp, m.Kd       = float(kp), float(kd)

def main():
    # Init
    udp   = sdk.UDP(LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    safe  = sdk.Safety(sdk.LeggedType.Go1)
    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    # Warmup
    for _ in range(10):
        udp.Recv(); udp.GetRecv(state); time.sleep(0.005)

    # Current posture
    q0 = [state.motorState[i].q for i in range(12)]
    # Load stand target (fallback to q0)
    if os.path.exists(STAND_FILE):
        qStand = np.load(STAND_FILE).astype(float).tolist()
        if len(qStand) != 12: qStand = list(q0)
    else:
        qStand = list(q0)

    print("► Phase A: 10s smooth ramp to stand (no snap).")
    steps = max(1, int(RAMP_TO_STAND_S / DT))
    tA = time.time()
    for s in range(steps):
        tic = time.time()
        a = s_curve((s+1)/steps)

        # Right legs → stand (supportive)
        for R in RIGHT_LEGS:
            for j in range(3):
                idx = d[f'{R}_{j}']
                q_cmd = (1-a)*q0[idx] + a*qStand[idx]
                set_joint(cmd, idx, q_cmd, KP_R, KD_R)

        # Left legs → stand (use "move" gains so never rigid during ramp)
        for L in LEFT_LEGS:
            # hip (0), thigh (1)
            for j in (0,1):
                idx = d[f'{L}_{j}']
                q_cmd = (1-a)*q0[idx] + a*qStand[idx]
                set_joint(cmd, idx, q_cmd, KP_L_MOVE_HIPTHIGH, KD_L_MOVE_HIPTHIGH)
            # knee (2)
            idx = d[f'{L}_2']
            q_cmd = (1-a)*q0[idx] + a*qStand[idx]
            set_joint(cmd, idx, q_cmd, KP_L_MOVE_KNEE, KD_L_MOVE_KNEE)

        if (time.time() - tA) > 1.0:
            safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd); udp.Send()
        time.sleep(max(0.0, DT - (time.time() - tic)))

    print(f"► Phase B: Teach & firm-hold window ({WAIT_BEFORE_RECORD_S:.0f}s). Move LEFT legs; release to latch.")

    # Setup teach-and-hold
    left_joint_indices = []
    for L in LEFT_LEGS:
        left_joint_indices += [d[f'{L}_0'], d[f'{L}_1'], d[f'{L}_2']]

    target     = {idx: qStand[idx] for idx in left_joint_indices}
    latched    = {idx: False for idx in left_joint_indices}
    last_move  = {idx: time.time() for idx in left_joint_indices}
    boost_until= {idx: 0.0 for idx in left_joint_indices}

    tB_start = time.time()
    last_print = -1

    while True:
        tic = time.time()
        udp.Recv(); udp.GetRecv(state)

        # Right legs: hold stand supportively
        for R in RIGHT_LEGS:
            for j in range(3):
                idx = d[f'{R}_{j}']
                set_joint(cmd, idx, qStand[idx], KP_R, KD_R)

        # Left legs: teach & hold per joint
        for L in LEFT_LEGS:
            # HIP (0)
            hip = d[f'{L}_0']
            qh = state.motorState[hip].q
            dqh= state.motorState[hip].dq
            if abs(dqh) > VEL_THRESH:
                # moving: follow easy
                latched[hip] = False
                last_move[hip] = time.time()
                target[hip] = qh
                set_joint(cmd, hip, qh, KP_L_MOVE_HIPTHIGH, KD_L_MOVE_HIPTHIGH, 
                          tau=HIP_BIAS_SIGN*HIP_BIAS_TAU)  # bias outward while moving
            else:
                # still: latch after dwell
                if not latched[hip] and (time.time()-last_move[hip]) >= DWELL_SEC:
                    latched[hip] = True
                    target[hip]  = qh
                # anti-drift
                err = target[hip] - qh
                kp = KP_L_HOLD_HIPTHIGH
                kd = KD_L_HOLD_HIPTHIGH
                tau = HIP_BIAS_SIGN*HIP_BIAS_TAU
                if abs(err) > ERR_DEADBAND:
                    kp = KP_L_HOLD_HIPTHIGH + KP_BOOST
                    boost_until[hip] = time.time()+BOOST_TIME
                elif time.time() < boost_until[hip]:
                    kp = KP_L_HOLD_HIPTHIGH + KP_BOOST
                set_joint(cmd, hip, target[hip], kp, kd, tau=tau)

            # THIGH (1)
            th = d[f'{L}_1']
            qt = state.motorState[th].q
            dqt= state.motorState[th].dq
            if abs(dqt) > VEL_THRESH:
                latched[th] = False; last_move[th] = time.time(); target[th] = qt
                set_joint(cmd, th, qt, KP_L_MOVE_HIPTHIGH, KD_L_MOVE_HIPTHIGH)
            else:
                if not latched[th] and (time.time()-last_move[th]) >= DWELL_SEC:
                    latched[th] = True; target[th] = qt
                err = target[th] - qt
                kp = KP_L_HOLD_HIPTHIGH; kd = KD_L_HOLD_HIPTHIGH
                if abs(err) > ERR_DEADBAND:
                    kp = KP_L_HOLD_HIPTHIGH + KP_BOOST
                    boost_until[th] = time.time()+BOOST_TIME
                elif time.time() < boost_until[th]:
                    kp = KP_L_HOLD_HIPTHIGH + KP_BOOST
                set_joint(cmd, th, target[th], kp, kd)

            # KNEE (2)
            kn = d[f'{L}_2']
            qk = state.motorState[kn].q
            dqk= state.motorState[kn].dq
            if abs(dqk) > VEL_THRESH:
                latched[kn] = False; last_move[kn] = time.time(); target[kn] = qk
                set_joint(cmd, kn, qk, KP_L_MOVE_KNEE, KD_L_MOVE_KNEE)
            else:
                if not latched[kn] and (time.time()-last_move[kn]) >= DWELL_SEC:
                    latched[kn] = True; target[kn] = qk
                err = target[kn] - qk
                kp = KP_L_HOLD_KNEE; kd = KD_L_HOLD_KNEE
                if abs(err) > ERR_DEADBAND:
                    kp = KP_L_HOLD_KNEE + KP_BOOST
                    boost_until[kn] = time.time()+BOOST_TIME
                elif time.time() < boost_until[kn]:
                    kp = KP_L_HOLD_KNEE + KP_BOOST
                set_joint(cmd, kn, target[kn], kp, kd)

        safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd); udp.Send()

        # Countdown
        elapsed = time.time() - tB_start
        remaining = int(max(0, WAIT_BEFORE_RECORD_S - elapsed))
        if remaining % 5 == 0 and remaining != last_print:
            print(f"… recording in {remaining}s")
            last_print = remaining
        if elapsed >= WAIT_BEFORE_RECORD_S:
            break

        time.sleep(max(0.0, DT - (time.time() - tic)))

    # Phase C: record pose + IMU
    udp.Recv(); udp.GetRecv(state)
    pose = np.array([state.motorState[i].q for i in range(12)], dtype=float)
    rpy  = np.array(state.imu.rpy, dtype=float)
    ts = time.strftime("%Y%m%d-%H%M%S")
    np.save(f"recorded_pose_{ts}.npy", pose)
    np.save(f"recorded_orientation_rpy_{ts}.npy", rpy)
    print("► Recording pose now…")
    print(f"✓ Saved joint pose → recorded_pose_{ts}.npy")
    print(f"✓ Saved IMU RPY   → recorded_orientation_rpy_{ts}.npy")

    # Release
    for i in range(12):
        m = cmd.motorCmd[i]
        m.q, m.dq, m.tau = 0.0, 0.0, 0.0
        m.Kp, m.Kd       = 0.0, 0.0
    udp.SetSend(cmd); udp.Send()
    print("Done.")

if __name__ == "__main__":
    main()
