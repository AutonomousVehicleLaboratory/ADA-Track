#!/usr/bin/env python

import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped
from vision_msgs.msg import Detection3DArray

class ADATrackNode:
    def __init__(self):
        rospy.init_node('transform_subscriber')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.detection_sub = rospy.Subscriber('/detection3D', Detection3DArray, self.detection_callback)
        
    def get_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn(f"Failed to lookup transform from {source_frame} to {target_frame}")
            return None
    
    def transform_to_matrices(self, transform):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        
        # Translation matrix
        trans_matrix = np.array([
            [1, 0, 0, translation.x],
            [0, 1, 0, translation.y],
            [0, 0, 1, translation.z],
            [0, 0, 0, 1]
        ])
        
        # Rotation matrix
        qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
        rot_matrix = np.array([
            [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw, 0],
            [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw, 0],
            [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2, 0],
            [0, 0, 0, 1]
        ])
        
        return trans_matrix, rot_matrix
    
    def detection_callback(self, msg):
        lidar_to_world = self.get_transform('world', 'lidar')
        lidar_to_image = self.get_transform('image', 'lidar')
        
        if lidar_to_world and lidar_to_image:
            l2w_trans, l2w_rot = self.transform_to_matrices(lidar_to_world)
            l2i_trans, l2i_rot = self.transform_to_matrices(lidar_to_image)
            
            rospy.loginfo("Lidar to World Translation:\n%s", l2w_trans)
            rospy.loginfo("Lidar to World Rotation:\n%s", l2w_rot)
            rospy.loginfo("Lidar to Image Translation:\n%s", l2i_trans)
            rospy.loginfo("Lidar to Image Rotation:\n%s", l2i_rot)
            
            # Process Detection3DArray message here
            # ...

if __name__ == '__main__':
    try:
        node = TransformSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
