#!/usr/bin/python3
# ------------------------------------------------------------
# 00_lock_front.py  --  Holds joints 0-5 rigid so front paws
# stay glued to skateboard.  Ctrl-C to release.
# ------------------------------------------------------------
import sys, time, pathlib
SDK_PY = pathlib.Path(__file__).resolve().parents[2] / "lib/python/amd64"
sys.path.append(str(SDK_PY))
import robot_interface as sdk

LOWLEVEL = 0xff
DT       = 0.002
front_ids = [0,1,2, 3,4,5]

udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
cmd  = sdk.LowCmd();  state = sdk.LowState()
udp.InitCmdData(cmd)

fixed = [0.0]*6
captured = False
print("▶  Front-leg lock running.  Ctrl-C to exit.")

try:
    while True:
        time.sleep(DT)
        udp.Recv(); udp.GetRecv(state)

        if not captured:                       # capture once
            for i,m in enumerate(front_ids):
                fixed[i] = state.motorState[m].q
            captured = True
            print("Angles frozen:", [round(a,3) for a in fixed])

        for i,m in enumerate(front_ids):       # stiff PD
            mc = cmd.motorCmd[m]
            mc.q, mc.dq  = fixed[i], 0.0
            mc.Kp, mc.Kd = 25.0, 2.0
            mc.tau       = 0.0

        udp.SetSend(cmd); udp.Send()

except KeyboardInterrupt:
    for mc in cmd.motorCmd: mc.Kp = mc.Kd = mc.tau = 0
    udp.SetSend(cmd); udp.Send()
    print("\nFront-leg lock released.")

