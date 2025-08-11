#!/usr/bin/python
#
# save_position.py  – capture *any* pose(s) you like, one after another.
#
# Usage:
#   • Robot in HIGH-LEVEL mode, holding the pose you want.
#   • python3 save_position.py
#   • Enter a label (e.g. "sit"), press ENTER → script averages 1 s of data.
#   • Repeat with new labels; press ENTER on a blank prompt to exit.
#
import sys, time, json, numpy as np, os
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# ------------- config -------------
HIGHLEVEL  = 0xee        # just listen; no commands sent
UDP_PORT   = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
SAMPLES    = 500         # 500 × 2 ms ≈ 1 s averaging
# ----------------------------------

# joint-name → index table (same order as SDK examples)
d = {f'{leg}_{j}': i
     for i,(leg,j) in enumerate([(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}

udp   = sdk.UDP(HIGHLEVEL, UDP_PORT, ROBOT_IP, ROBOT_PORT)
state = sdk.HighState()

print("\n=== Pose Recorder ===")
print("Robot should be in the desired posture, balanced by its own controller.")
print("Enter a pose name and press <Enter> to capture; empty input quits.\n")

while True:
    pose_name = input("Pose name (blank to exit): ").strip()
    if pose_name == "":
        print("Finished recording poses.")
        break

    # avoid overwriting accidentally
    npy_file  = f"{pose_name}_pose.npy"
    json_file = f"{pose_name}_pose.json"
    if os.path.exists(npy_file) or os.path.exists(json_file):
        ans = input(f"Files for '{pose_name}' exist. Overwrite? [y/N] ").lower()
        if ans != "y":
            print("Skipping.")
            continue

    print(f"► Capturing '{pose_name}' … please keep the robot still for 1 s.")
    ang_buf = []
    for _ in range(SAMPLES):
        time.sleep(0.002)
        udp.Recv(); udp.GetRecv(state)
        ang_buf.append([state.motorState[i].q for i in range(12)])

    pose = np.mean(np.array(ang_buf), axis=0)  # 12-vector

    np.save(npy_file, pose)
    with open(json_file, "w") as f:
        json.dump({k: float(pose[i]) for k, i in d.items()}, f, indent=2)

    print(f"✓ Saved {npy_file} and {json_file}\n")
