#!/usr/bin/python
#
# save_side_pose.py
#
# 1. Arrange the robot so the two “side” legs are on the ground
#    and the other two legs are on the skateboard, still controlled
#    by HIGH-LEVEL balance.  Keep the pose steady.
# 2. Run:  python3 save_side_pose.py
# 3. It averages 1 s of joint data and saves side_pose.{npy,json}.
#
import sys, time, json, numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

HIGHLEVEL  = 0xee
UDP_PORT   = 8080
ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
SAMPLES    = 500               # 1 s @ 500 Hz

# joint-name → index (FR_0 … RL_2)
d = {f'{leg}_{j}': i for i,(leg,j) in enumerate(
     [(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}

udp   = sdk.UDP(HIGHLEVEL, UDP_PORT, ROBOT_IP, ROBOT_PORT)
state = sdk.HighState()

print("► Capturing side-pose… keep the robot still for 1 s.")
buf = []
for _ in range(SAMPLES):
    time.sleep(0.002)
    udp.Recv();  udp.GetRecv(state)
    buf.append([state.motorState[i].q for i in range(12)])

pose = np.mean(np.array(buf), axis=0)

np.save('side_pose.npy', pose)
with open('side_pose.json', 'w') as f:
    json.dump({k: float(pose[i]) for k,i in d.items()}, f, indent=2)

print("✓ Saved side_pose.npy / side_pose.json")
