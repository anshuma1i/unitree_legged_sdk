#!/usr/bin/python
#
# Run this while the robot is balancing in HIGH-LEVEL mode.
# It listens only, averages 1 s of joint angles, and writes
#    stand_pose.npy          (NumPy 12-vector)
#    stand_pose.json         (human-readable)
#
import sys, time, json, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

HIGHLEVEL = 0xee           # listen on the HL broadcast
UDP_PORT  = 8080
ROBOT_IP  = "192.168.123.10"
ROBOT_PORT= 8007

SAMPLES   = 500            # 500 × 2 ms  ≈ 1 s averaging window

d = {f'{leg}_{j}': i for i,(leg,j) in enumerate(
     [(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}

udp   = sdk.UDP(HIGHLEVEL, UDP_PORT, ROBOT_IP, ROBOT_PORT)
state = sdk.HighState()    # any state msg works for read-only
ang_buf = []

print("► Collecting joint angles for 1 s … stand still.")
for _ in range(SAMPLES):
    time.sleep(0.002)
    udp.Recv();  udp.GetRecv(state)
    ang_buf.append([state.motorState[i].q for i in range(12)])

pose = np.mean(np.array(ang_buf), axis=0)        # 12-vector
np.save('stand_pose.npy', pose)
with open('stand_pose.json', 'w') as f:
    json.dump({k: float(pose[i]) for k,i in d.items()}, f, indent=2)

print("✓ Saved stand_pose.npy / .json – you can hang the robot now.")
