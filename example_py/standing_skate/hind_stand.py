#!/usr/bin/python
#
# save_hind_pose.py
#
# 1. Put the robot in HIGH‑LEVEL mode, balanced with two legs on
#    the ground and the other two on the skateboard.
# 2. Run:  python3 save_hind_pose.py
# 3. After 1 s the script stores hind_pose.npy / .json in cwd.
#
import sys, time, json, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

HIGHLEVEL   = 0xee          # listen‑only
UDP_PORT    = 8080
ROBOT_IP    = "192.168.123.10"
ROBOT_PORT  = 8007
SAMPLES     = 500           # 500 × 2 ms ≈ 1 s

# joint‑name → index (FR_0 … RL_2)
d = {f'{leg}_{j}': i for i,(leg,j) in enumerate(
     [(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}

udp   = sdk.UDP(HIGHLEVEL, UDP_PORT, ROBOT_IP, ROBOT_PORT)
state = sdk.HighState()

print("► Capture starting – keep the hind‑pose steady for 1 s.")
buf = []
for _ in range(SAMPLES):
    time.sleep(0.002)
    udp.Recv();  udp.GetRecv(state)
    buf.append([state.motorState[i].q for i in range(12)])

pose = np.mean(np.array(buf), axis=0)      # 12‑vector

np.save('hind_pose.npy', pose)
with open('hind_pose.json', 'w') as f:
    json.dump({k: float(pose[i]) for k,i in d.items()}, f, indent=2)

print("✓ Saved hind_pose.npy / hind_pose.json")
