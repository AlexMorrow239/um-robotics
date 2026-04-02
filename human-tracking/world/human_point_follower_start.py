import os
import signal
import subprocess
import time

from process_handler import process_handler

PACKAGE_NAME = "human_point_follower"
PACKAGE_PATH = subprocess.getoutput(f"rospack find {PACKAGE_NAME}")
print(f"Package Path: {PACKAGE_PATH}")

if 'error' in PACKAGE_PATH.lower():
    print(f'source and run program again!')
    exit()

LAUNCH_NAME = 'pointing_detector.launch'
LAUNCH_PATH = os.path.join(PACKAGE_PATH, 'launch', LAUNCH_NAME)
print(f'roslaunch path:', LAUNCH_PATH)

ISAAC_SIM_NAME = "isaac_sim"
ISAAC_SIM_PATH = subprocess.getoutput(f"rospack find {ISAAC_SIM_NAME}")
print(f"Isaac Sim path: {ISAAC_SIM_PATH}")

ISAAC_SIM_PYTHON_NAME = 'python.sh'
ISAAC_SIM_PYTHON_PATH = os.path.join(ISAAC_SIM_PATH, ISAAC_SIM_PYTHON_NAME)
print(f'Isaac Sim python path:', ISAAC_SIM_PYTHON_PATH)

ISAAC_WORLD_NAME = 'human_point_follower_world.py'
ISAAC_WORLD_PATH = os.path.join(PACKAGE_PATH, 'world', ISAAC_WORLD_NAME)
print(f'Isaac Sim world path:', ISAAC_WORLD_PATH)

RVIZ_NAME = 'pointing_detector.rviz'
RVIZ_PATH = os.path.join(PACKAGE_PATH, 'rviz', RVIZ_NAME)
print(f'RVIZ path:', RVIZ_PATH)

roslaunch_cmd = f"roslaunch {LAUNCH_PATH}"
isaac_sim_cmd = f"{ISAAC_SIM_PYTHON_PATH} {ISAAC_WORLD_PATH}"
rviz_cmd = f"rviz -d {RVIZ_PATH}"

# Start processes
processes = {
    "roslaunch": process_handler.start_process(roslaunch_cmd, capture_output=False),
    "isaac_sim": process_handler.start_process(isaac_sim_cmd, capture_output=True),
}

signal.signal(signal.SIGINT, lambda signum, frame: process_handler.cleanup_exit(processes))
signal.signal(signal.SIGTERM, lambda signum, frame: process_handler.cleanup_exit(processes))

# Wait indefinitely
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    process_handler.cleanup_exit(processes=processes)