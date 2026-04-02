import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from tf import TransformListener

class TFHelper:
    def __init__(self):
        self.tf_listener = TransformListener()

    def transform_point(self, point_position, source_frame, target_frame):
        point_posestamped = PoseStamped()
        point_posestamped.header.frame_id = source_frame
        point_posestamped.header.stamp = rospy.Time(0)

        point_posestamped.pose.position.x = point_position[0]
        point_posestamped.pose.position.y = point_position[1]
        point_posestamped.pose.position.z = point_position[2]

        self.tf_listener.lookupTransform(target_frame=target_frame,
                                         source_frame=source_frame,
                                         time=rospy.Time(0))
        
        transformed_pose = self.tf_listener.transformPose(target_frame=target_frame,
                                                          ps=point_posestamped)
        
        transformed_point_position = np.array([transformed_pose.pose.position.x, transformed_pose.pose.position.y, transformed_pose.pose.position.z])

        return transformed_point_position
    