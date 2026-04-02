#!/usr/bin/env python3

import sys
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray, Pose, Point
from cv_bridge import CvBridge

# Add OpenPose Python path (set OPENPOSE_PYTHON_PATH env var or update this default)
openpose_python_path = os.environ.get('OPENPOSE_PYTHON_PATH', '/opt/openpose/build/python')
sys.path.append(openpose_python_path)
from openpose import pyopenpose as op

class OpenPoseOmniverseBridge:
    def __init__(self):
        rospy.init_node('openpose_omniverse_bridge')
        
        # Load parameters from yaml
        model_folder = rospy.get_param('/openpose/model_folder')
        model_pose = rospy.get_param('/openpose/model_pose', 'BODY_25')
        net_resolution = rospy.get_param('/openpose/net_resolution', '320x176')
        
        # Setup OpenPose
        params = dict()
        params["model_folder"] = model_folder
        params["model_pose"] = model_pose
        params["net_resolution"] = net_resolution
        
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        rospy.loginfo("OpenPose initialized!")
        
        # Setup ROS
        self.bridge = CvBridge()
        self.min_keypoints = rospy.get_param('/human_detection/min_keypoints', 10)
        
        # Subscribe to camera
        camera_topic = rospy.get_param('/camera/rgb_topic')
        self.image_sub = rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1)
        
        # Publishers
        self.detection_pub = rospy.Publisher('/openpose/human_detected', Bool, queue_size=1)
        self.skeleton_image_pub = rospy.Publisher('/openpose/skeleton_image', Image, queue_size=1)
        self.keypoints_pub = rospy.Publisher('/openpose/keypoints', PoseArray, queue_size=1)

        # BODY_25 keypoint indices
        self.NOSE_IDX = 0
        self.NECK_IDX = 1
        self.MIDHIP_IDX = 8

        # Arm keypoint indices for pointing detection
        self.RSHOULDER_IDX = 2
        self.RELBOW_IDX = 3
        self.RWRIST_IDX = 4
        self.LSHOULDER_IDX = 5
        self.LELBOW_IDX = 6
        self.LWRIST_IDX = 7

        # Publisher for arm keypoints (used by pointing detector)
        self.arm_keypoints_pub = rospy.Publisher('/openpose/arm_keypoints', PoseArray, queue_size=1)

        rospy.loginfo(f"Subscribed to: {camera_topic}")
        
    def image_callback(self, msg):
        try:
            # Convert and process
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            datum = op.Datum()
            datum.cvInputData = cv_image
            self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            # Check detection
            human_detected = False
            keypoint_array = PoseArray()
            keypoint_array.header.stamp = msg.header.stamp
            keypoint_array.header.frame_id = "camera_rgb_optical_frame"

            if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                valid_keypoints = sum(1 for kp in datum.poseKeypoints[0] if kp[2] > 0.3)
                human_detected = valid_keypoints >= self.min_keypoints

                if human_detected:
                    rospy.loginfo_throttle(1.0, f"Human detected with {valid_keypoints} keypoints")

                    # Extract Neck and MidHip keypoints for tracking
                    keypoints = datum.poseKeypoints[0]

                    # Neck keypoint
                    if len(keypoints) > self.NECK_IDX and keypoints[self.NECK_IDX][2] > 0.3:
                        neck_pose = Pose()
                        neck_pose.position.x = keypoints[self.NECK_IDX][0]  # pixel x
                        neck_pose.position.y = keypoints[self.NECK_IDX][1]  # pixel y
                        neck_pose.position.z = keypoints[self.NECK_IDX][2]  # confidence
                        keypoint_array.poses.append(neck_pose)

                    # MidHip keypoint
                    if len(keypoints) > self.MIDHIP_IDX and keypoints[self.MIDHIP_IDX][2] > 0.3:
                        midhip_pose = Pose()
                        midhip_pose.position.x = keypoints[self.MIDHIP_IDX][0]  # pixel x
                        midhip_pose.position.y = keypoints[self.MIDHIP_IDX][1]  # pixel y
                        midhip_pose.position.z = keypoints[self.MIDHIP_IDX][2]  # confidence
                        keypoint_array.poses.append(midhip_pose)

                    # Publish arm keypoints for pointing detection
                    # Order: Nose, Neck, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist
                    # Head keypoints added for face-to-hand pointing vector (per paper method)
                    arm_keypoint_array = PoseArray()
                    arm_keypoint_array.header.stamp = msg.header.stamp
                    arm_keypoint_array.header.frame_id = "camera_rgb_optical_frame"

                    arm_indices = [
                        self.NOSE_IDX, self.NECK_IDX,  # Head keypoints for pointing origin
                        self.LSHOULDER_IDX, self.LELBOW_IDX, self.LWRIST_IDX,  # Left arm (priority)
                        self.RSHOULDER_IDX, self.RELBOW_IDX, self.RWRIST_IDX   # Right arm (fallback)
                    ]

                    for idx in arm_indices:
                        arm_pose = Pose()
                        if len(keypoints) > idx:
                            arm_pose.position.x = keypoints[idx][0]  # pixel x
                            arm_pose.position.y = keypoints[idx][1]  # pixel y
                            arm_pose.position.z = keypoints[idx][2]  # confidence
                        else:
                            arm_pose.position.x = 0
                            arm_pose.position.y = 0
                            arm_pose.position.z = 0  # confidence 0 = not detected
                        arm_keypoint_array.poses.append(arm_pose)

                    self.arm_keypoints_pub.publish(arm_keypoint_array)

            # Publish results
            self.detection_pub.publish(Bool(human_detected))
            self.keypoints_pub.publish(keypoint_array)
            
            if datum.cvOutputData is not None:
                skeleton_msg = self.bridge.cv2_to_imgmsg(datum.cvOutputData, "bgr8")
                self.skeleton_image_pub.publish(skeleton_msg)
                
        except Exception as e:
            rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    try:
        bridge = OpenPoseOmniverseBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass