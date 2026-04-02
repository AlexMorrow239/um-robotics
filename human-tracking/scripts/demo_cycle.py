#!/usr/bin/env python3
"""
Demo Cycle Script

Cycles through arm poses and runs the human point follower demo.
Waits for user input (Enter key) before each step so you can
position the character model in Isaac Sim.

Usage:
    rosrun human_point_follower demo_cycle.py
"""

import subprocess
import time

# All arm poses to cycle through
ARM_POSES = [
    "right_point",
    "left_point_far",
    "left_point_lateral",
]

def publish_arm_pose(pose_name):
    """Publish arm pose via rostopic."""
    cmd = ["rostopic", "pub", "-1", "/human/arm_pose", "std_msgs/String", f"data: '{pose_name}'"]
    subprocess.run(cmd, capture_output=True)
    print(f"  Published arm pose: {pose_name}")

def start_demo():
    """Start the human point follower demo."""
    cmd = ["rostopic", "pub", "-1", "/human_point_follower/start", "std_msgs/Bool", "data: true"]
    subprocess.run(cmd, capture_output=True)
    print("  Started demo")

def reset_demo():
    """Reset the demo back to IDLE."""
    cmd = ["rostopic", "pub", "-1", "/human_point_follower/start", "std_msgs/Bool", "data: true"]
    subprocess.run(cmd, capture_output=True)
    print("  Reset demo to IDLE")

def get_current_state():
    """Get current state from the state topic."""
    try:
        result = subprocess.run(
            ["rostopic", "echo", "-n", "1", "/human_point_follower/state"],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if result.returncode == 0:
            # Parse "data: STATE_NAME" format
            for line in result.stdout.strip().split('\n'):
                if 'data:' in line:
                    return line.split('"')[1] if '"' in line else line.split(':')[1].strip()
        return "UNKNOWN"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def wait_for_completion(timeout=120):
    """Wait for demo to complete (reach COMPLETED state)."""
    print("  Waiting for demo to complete...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        state = get_current_state()
        if state == "COMPLETED":
            print("  Demo completed!")
            return True
        elif state == "IDLE":
            # Demo was reset externally
            print("  Demo was reset")
            return True
        time.sleep(1.0)
    print(f"  Timeout after {timeout}s")
    return False

def main():
    print("=" * 60)
    print("Human Point Follower Demo Cycle")
    print("=" * 60)
    print()
    print("This script will cycle through all arm poses and run the demo.")
    print("Press Enter to advance through each step.")
    print()
    print(f"Arm poses to test: {len(ARM_POSES)}")
    for i, pose in enumerate(ARM_POSES):
        print(f"  {i+1}. {pose}")
    print()

    input("Press Enter to begin...")
    print()

    for i, pose in enumerate(ARM_POSES):
        print("=" * 60)
        print(f"POSE {i+1}/{len(ARM_POSES)}: {pose}")
        print("=" * 60)
        print()

        # Step 1: Set arm pose
        print(f"[Step 1] Setting arm pose to '{pose}'")
        input("  Press Enter to set arm pose...")
        publish_arm_pose(pose)
        print()

        # Step 2: Position character (user does this manually)
        print("[Step 2] Position the character model in Isaac Sim if needed")
        input("  Press Enter when ready to start demo...")
        print()

        # Step 3: Start demo
        print("[Step 3] Starting demo")
        start_demo()
        print()

        # Step 4: Wait for completion
        print("[Step 4] Demo running...")
        wait_for_completion()
        print()

        # Step 5: Reset for next pose
        if i < len(ARM_POSES) - 1:
            print("[Step 5] Resetting for next pose")
            input("  Press Enter to reset and continue to next pose...")
            reset_demo()
            time.sleep(1.0)  # Give time for reset
            print()
        else:
            print("All poses completed!")

    print()
    print("=" * 60)
    print("Demo cycle complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
