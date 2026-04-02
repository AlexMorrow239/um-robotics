import rospy

from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

class RvizHelper:
    def __init__(self): pass

    def generate_marker(self, 
                        ns, 
                        frame_id, 
                        marker_type, 
                        marker_action, 
                        marker_id, 
                        marker_scale, 
                        marker_color,
                        marker_text=None,
                        marker_position=None,
                        marker_orientation=None,
                        marker_points=None):
        marker = Marker()
        marker.ns = ns
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time(0)

        marker.type = marker_type
        marker.action = marker_action
        marker.id = marker_id
        
        marker.scale.x = marker_scale[0]
        marker.scale.y = marker_scale[1]
        marker.scale.z = marker_scale[2]

        marker.color.r = marker_color[0]
        marker.color.g = marker_color[1]
        marker.color.b = marker_color[2]
        marker.color.a = marker_color[3]

        if marker_text is not None:
            marker.text = marker_text

        if marker_position is not None:
            marker.pose.position.x = marker_position[0]
            marker.pose.position.y = marker_position[1]
            marker.pose.position.z = marker_position[2]

        if marker_orientation is not None:
            marker_orientation_quaternion = quaternion_from_euler(marker_orientation[0], marker_orientation[1], marker_orientation[2])

        else:
            marker_orientation_quaternion = quaternion_from_euler(0, 0, 0)

        marker.pose.orientation.x = marker_orientation_quaternion[0]
        marker.pose.orientation.y = marker_orientation_quaternion[1]
        marker.pose.orientation.z = marker_orientation_quaternion[2]
        marker.pose.orientation.w = marker_orientation_quaternion[3]

        if marker_points is not None:
            for point in marker_points:
                marker_point = Point(point[0], point[1], point[2])
                marker.points.append(marker_point)

        return marker

