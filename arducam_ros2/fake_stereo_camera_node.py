#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import random

class FakeStereoCameraNode(Node):
    def __init__(self):
        super().__init__('fake_stereo_camera_node')
        self.left_image_pub = self.create_publisher(Image, 'left/image_raw', 10)
        self.right_image_pub = self.create_publisher(Image, 'right/image_raw', 10)
        self.left_info_pub = self.create_publisher(CameraInfo, 'left/camera_info', 10)
        self.right_info_pub = self.create_publisher(CameraInfo, 'right/camera_info', 10)
        self.bridge = CvBridge()

        self.timer_period = 1.0 / 60.0  # 60 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    def timer_callback(self):
        # Generate left and right images
        left_image, left_info = self.generate_random_test_pattern_info(320, 240)
        right_image, right_info = self.generate_test_pattern_info(320, 240)

        # Convert images to ROS Image messages
        left_image_msg = self.bridge.cv2_to_imgmsg(left_image, encoding='bgr8')
        right_image_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='bgr8')

        left_info.header.stamp = self.get_clock().now().to_msg()
        right_info.header.stamp = self.get_clock().now().to_msg()

        # Publish left and right images
        self.left_image_pub.publish(left_image_msg)
        self.left_info_pub.publish(left_info)

        self.right_image_pub.publish(right_image_msg)
        self.right_info_pub.publish(right_info)

    def generate_test_pattern(self, width, height):
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a test pattern (you can modify this to create different patterns)
        cv2.rectangle(pattern, (20, 20), (width - 20, height - 20), (255, 255, 255), -1)
        return pattern

    def generate_random_test_pattern_info(self, width, height):
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw random rectangles within the image
        num_rectangles = random.randint(3, 7)  # Choose random number of rectangles
        for _ in range(num_rectangles):
            # Generate random coordinates for the top-left and bottom-right corners of the rectangle
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(x1 + 1, width)
            y2 = random.randint(y1 + 1, height)
            
            # Generate random color for the rectangle
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Draw the rectangle
            cv2.rectangle(pattern, (x1, y1), (x2, y2), color, -1)
        
        # Create CameraInfo message
        
        info = CameraInfo()
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients
        info.k = [300.0, 0.0, width / 2.0, 0.0, 300.0, height / 2.0, 0.0, 0.0, 1.0]  # Camera matrix
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Rectification matrix
        info.p = [300.0, 0.0, width / 2.0, 0.0, 0.0, 300.0, height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Projection matrix

        return pattern ,info

    '''def generate_test_patterns(self):
        while True:
            # left_image, left_info = self.generate_test_pattern(320, 240)
            # right_image, right_info = self.generate_test_pattern(320, 240)
            left_image = self.generate_test_pattern(320, 240)
            right_image = self.generate_test_pattern(320, 240)

            left_image_msg = self.bridge.cv2_to_imgmsg(left_image, encoding='bgr8')
            right_image_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='bgr8')

            # left_info.header.stamp = self.get_clock().now().to_msg()
            # right_info.header.stamp = self.get_clock().now().to_msg()

            self.left_image_pub.publish(left_image_msg)
            self.right_image_pub.publish(right_image_msg)

            # self.left_info_pub.publish(left_info)
            # self.right_info_pub.publish(right_info)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break'''

    def generate_test_pattern_info(self, width, height):
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a test pattern (you can modify this to create different patterns)
        # Generate random color for the rectangle
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(pattern, (20, 20), (width - 20, height - 20), color, -1)

        # Create CameraInfo message
        
        info = CameraInfo()
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients
        info.k = [300.0, 0.0, width / 2.0, 0.0, 300.0, height / 2.0, 0.0, 0.0, 1.0]  # Camera matrix
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Rectification matrix
        info.p = [300.0, 0.0, width / 2.0, 0.0, 0.0, 300.0, height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Projection matrix
        
        return pattern , info
        

def main(args=None):
    rclpy.init(args=args)
    fake_stereo_camera_node = FakeStereoCameraNode()
    # stereo_camera_node.generate_test_patterns()
    rclpy.spin(fake_stereo_camera_node)
    fake_stereo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
