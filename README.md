# Robotics Portfolio

ROS-based robotics projects using the Toyota HSR (Human Support Robot) in NVIDIA Isaac Sim environments. Covers localization, path planning, and human-robot interaction.

> **Note:** These projects depend on private University of Miami Robotics and Toyota HSR simulation environments and are not publicly runnable. Code is provided for reference and portfolio purposes.

**Built with:** Python · ROS · NVIDIA Isaac Sim · OpenPose · Numba · NumPy · SciPy · OpenCV

## Projects

### [Localization & Navigation](localization-navigation/)

Particle filter localization and autonomous navigation for the Toyota HSR in a simulated lab environment. Includes map processing, odometry correction, and RViz visualization.

### [RRT Path Planning](rrt-path-planning/)

Rapidly-exploring Random Tree (RRT) based path planning and navigation for the HSR robot, with obstacle avoidance in a simulated environment.

### [Human Tracking & Gesture Recognition](human-tracking/)

Human tracking and pointing gesture recognition on the HSR platform. Uses OpenPose skeleton detection and a face-to-hand 3D pointing vector to identify target locations, then navigates toward them.

## Acknowledgments

These projects were developed as part of CSC 752 (Robotics) at the University of Miami. Taught by Ubbo Visser.
