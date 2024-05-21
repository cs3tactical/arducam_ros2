#!/usr/bin/env python

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import argparse
import subprocess
from .utils import ArducamUtils
import sys
import v4l2

import numpy as np
import random
import time

qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # Ensure this matches the subscriber
            durability=QoSDurabilityPolicy.VOLATILE,    # Ensure this matches the subscriber
            depth=10  # Ensure this matches the subscriber
        )

class StereoCameraNode(Node):
    def __init__(self):
        super().__init__('stereo_camera_node')
        
        self.declare_parameter('device', 0)
        self._device = self.get_parameter('device').get_parameter_value().integer_value

        self.declare_parameter('pixelformat', 'Y16')
        self._pixelformat = self.get_parameter('pixelformat').get_parameter_value().string_value
        
        self.declare_parameter('width', 3840) # width of 2 images (concatenated horizontally)
        self._width = self.get_parameter('width').get_parameter_value().integer_value

        self.declare_parameter('height', 1200) # height of each image
        self._height = self.get_parameter('height').get_parameter_value().integer_value

        self.declare_parameter('fps', 37) # fps
        self._fps = self.get_parameter('fps').get_parameter_value().integer_value

        self.declare_parameter('frame_id', 'cam0')
        self._frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.declare_parameter('exposure', 200)
        self._exposure = self.get_parameter('exposure').get_parameter_value().integer_value

        # Set to True if you use monochrome image sensors. Set if RGB
        self.declare_parameter('out_grey', False)
        self._out_grey = self.get_parameter('out_grey').get_parameter_value().bool_value

        self.stereo_image_pub = self.create_publisher(Image, 'stereo/image_raw', qos_profile= qos)
        self.left_image_pub = self.create_publisher(Image, 'left/image_raw', qos_profile= qos)
        self.right_image_pub = self.create_publisher(Image, 'right/image_raw', qos_profile= qos)
        self.left_info_pub = self.create_publisher(CameraInfo, 'left/camera_info', qos_profile= qos)
        self.right_info_pub = self.create_publisher(CameraInfo, 'right/camera_info', qos_profile= qos)
        self.bridge = CvBridge()

        # open camera
        self._cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)

        # Set device configs
        if not self.setDevice():
            exit(1)
        
        # This is needed to be able to set exposue
        ret, frame = self._cap.read()
        cmd_str = 'v4l2-ctl -d /dev/video{} -c exposure={}'.format(self._device, self._exposure)
        subprocess.call([cmd_str],shell=True)

        cmd_str = 'v4l2-ctl -d /dev/video{} -c frame_rate={}'.format(self._device, self._fps)
        subprocess.call([cmd_str],shell=True)
        ret, frame = self._cap.read()

        self.timer_period = 1.0 / 60.0  # 60 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def get_fourcc(self,cap) -> str:
        """Return the 4-letter string of the codec the camera is using."""
        return int(cap.get(cv2.CAP_PROP_FOURCC))\
                .to_bytes(4, byteorder=sys.byteorder).decode()

    def setDevice(self):        
        try:
            # set pixel format
            if not self._cap.set(cv2.CAP_PROP_FOURCC, self.pixelformat(self._pixelformat)):
                self.get_logger().error("[ArduCamNode::setDevice] Failed to set pixelformat {}".format(self._pixelformat))
                return False
            print("pixel format: ", self.get_fourcc(self._cap))
        except Exception as e:
            self.get_logger().error("[ArduCamNode::setDevice] Failed to set pixelformat {}".format(e))
            return False
        
        self._arducam_utils = ArducamUtils(self._device)

        self.show_info(self._arducam_utils)
        
        # turn off RGB conversion
        if self._arducam_utils.convert2rgb == 0:
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, self._arducam_utils.convert2rgb)
            self.get_logger().info("[ArduCamNode::setDevice] RGB conversion is turned OFF")

        # set width
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        except Exception as e:
            self.get_logger().error("[ArduCamNode::setDevice] Failed to set width {}".format(e))
            return False
        
        # set height
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        except Exception as e:
            self.get_logger().error("[ArduCamNode::setDevice] Failed to set height {}".format(e))
            return False

        # set fps
        try:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        except Exception as e:
            self.get_logger().error("[ArduCamNode::setDevice] Failed to set fps {}".format(e))
            return False

        return True

    def timer_callback(self):

        ret, frame = self._cap.read()
        if not ret:
            return
        capture_time = self.get_clock().now().to_msg()

        # Define the new dimensions
        # new_width = 960  # New width in pixels
        # new_height = 600  # New height in pixels

        #3840x600 = 2X 1920x600
        #1920x600 = 7.3 fps
        #1280x400 = 9.5 fps

        # Start time
        start_time = time.time()
        # frame = self._arducam_utils.convert(frame)
        frame = cv2.convertScaleAbs(frame, None, 256.0 / (1 << 16)) # Scales, computes absolute values and converts the result to 8-bit. 
        frame = frame.astype(np.uint8) # convert frame to uint8
        frame = cv2.cvtColor(frame, 47) #convert color from bayer to BGR

        # encoding = "bgr8" if len(frame.shape) == 3 and frame.shape[2] >= 3 else "mono8"
        
        left_image = frame[:, :self._width//2]
        right_image = frame[:, self._width//2:]

        # Get the original width and height
        original_height, original_width = left_image.shape[:2]

        # Calculate the new width and height (half of the original dimensions)
        new_width = original_width // 3
        new_height = original_height // 3

        # Resize the frame
        left_image = cv2.resize(left_image, (new_width, new_height))
        right_image = cv2.resize(right_image, (new_width, new_height))
  
        if self._out_grey:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

            left_image_msg = self.bridge.cv2_to_imgmsg(left_image, encoding="mono8")
            left_image_msg.header.frame_id = self._frame_id

            right_image_msg = self.bridge.cv2_to_imgmsg(right_image, encoding="mono8")
            right_image_msg.header.frame_id = self._frame_id
        else: # color

            left_image = self.gray_world(left_image)
            right_image = self.gray_world(right_image)

            left_image_msg = self.bridge.cv2_to_imgmsg(left_image, encoding="bgr8")
            left_image_msg.header.frame_id = self._frame_id

            right_image_msg = self.bridge.cv2_to_imgmsg(right_image, encoding="bgr8")
            right_image_msg.header.frame_id = self._frame_id
            

        # stereo_image_msg = self.bridge.cv2_to_imgmsg(resized_frame, encoding=encoding)
        # stereo_image_msg.header.frame_id = self._frame_id

        # End time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time for the three functions: {:.5f} seconds, rate: {:.5f} Hz".format(elapsed_time,1/elapsed_time))

        # stereo_image_msg.header.stamp = capture_time
        left_image_msg.header.stamp = capture_time
        right_image_msg.header.stamp = capture_time
        
        # self.stereo_image_pub.publish(stereo_image_msg)

        # Publish left and right images
        self.left_image_pub.publish(left_image_msg)
        # self.left_info_pub.publish(left_info)

        self.right_image_pub.publish(right_image_msg)
        # self.right_info_pub.publish(right_info)

    def fourcc(self, a, b, c, d):
        return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)

    def pixelformat(self, string):
        if len(string) != 3 and len(string) != 4:
            msg = "{} is not a pixel format".format(string)
            raise argparse.ArgumentTypeError(msg)
        if len(string) == 3:
            return self.fourcc(string[0], string[1], string[2], ' ')
        else:
            return self.fourcc(string[0], string[1], string[2], string[3])

    def show_info(self, arducam_utils):
        _, firmware_version = arducam_utils.read_dev(ArducamUtils.FIRMWARE_VERSION_REG)
        _, sensor_id = arducam_utils.read_dev(ArducamUtils.FIRMWARE_SENSOR_ID_REG)
        _, serial_number = arducam_utils.read_dev(ArducamUtils.SERIAL_NUMBER_REG)
        _, pixformat_index_reg = arducam_utils.read_dev(ArducamUtils.PIXFORMAT_INDEX_REG)
        _, pixformat_type_reg = arducam_utils.read_dev(ArducamUtils.PIXFORMAT_TYPE_REG)
        _, pixformat_order_reg = arducam_utils.read_dev(ArducamUtils.PIXFORMAT_ORDER_REG)
        print("Firmware Version: {}".format(firmware_version))
        print("Sensor ID: 0x{:04X}".format(sensor_id))
        print("Serial Number: 0x{:08X}".format(serial_number))
        print("pixformat_index_reg: 0x{:04X}".format(pixformat_index_reg))
        print("pixformat_type_reg: 0x{:04X}".format(pixformat_type_reg))
        print("pixformat_order_reg: 0x{:04X}".format(pixformat_order_reg))

    def gray_world(self, nimg):
        nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
        mu_g = np.average(nimg[1])
        nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
        nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
        return  nimg.transpose(1, 2, 0).astype(np.uint8)

    def max_white(self, nimg):
        if nimg.dtype==np.uint8:
            brightest=float(2**8)
        elif nimg.dtype==np.uint16:
            brightest=float(2**16)
        elif nimg.dtype==np.uint32:
            brightest=float(2**32)
        else:
            brightest==float(2**8)
        nimg = nimg.transpose(2, 0, 1)
        nimg = nimg.astype(np.int32)
        nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
        nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
        nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)
        return nimg.transpose(1, 2, 0).astype(np.uint8)

    def retinex(self, nimg):
        nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
        mu_g = nimg[1].max()
        nimg[0] = np.minimum(nimg[0]*(mu_g/float(nimg[0].max())),255)
        nimg[2] = np.minimum(nimg[2]*(mu_g/float(nimg[2].max())),255)
        return nimg.transpose(1, 2, 0).astype(np.uint8)

    def retinex_adjust(self, nimg):
        """
        from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
        """
        nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
        sum_r = np.sum(nimg[0])
        sum_r2 = np.sum(nimg[0]**2)
        max_r = nimg[0].max()
        max_r2 = max_r**2
        sum_g = np.sum(nimg[1])
        max_g = nimg[1].max()
        coefficient = np.linalg.solve(np.array([[sum_r2,sum_r],[max_r2,max_r]]),
                                    np.array([sum_g,max_g]))
        nimg[0] = np.minimum((nimg[0]**2)*coefficient[0] + nimg[0]*coefficient[1],255)
        sum_b = np.sum(nimg[1])
        sum_b2 = np.sum(nimg[1]**2)
        max_b = nimg[1].max()
        max_b2 = max_r**2
        coefficient = np.linalg.solve(np.array([[sum_b2,sum_b],[max_b2,max_b]]),
                                                np.array([sum_g,max_g]))
        nimg[1] = np.minimum((nimg[1]**2)*coefficient[0] + nimg[1]*coefficient[1],255)
        return nimg.transpose(1, 2, 0).astype(np.uint8)

    def retinex_with_adjust(self, nimg):
        return self.retinex_adjust(self.retinex(nimg))
    
def main(args=None):
    rclpy.init(args=args)
    stereo_camera_node = StereoCameraNode()
    rclpy.spin(stereo_camera_node)
    stereo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
