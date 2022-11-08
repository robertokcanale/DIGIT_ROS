from digit_interface import Digit
import cv2, rospy, time, sys
import numpy as np
from utils.contact_area_functions import *
from utils.sensor_functions import *
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from digit_ros.msg import contact_center


class DIGIT:
    def __init__ (self, SERIAL_NUMBER, SENSOR_NAME, FPS_CONFIG = Digit.STREAMS["QVGA"]["fps"]["60fps"], intensity:int = 10):
        self.sn = SERIAL_NUMBER         
        self.name = SENSOR_NAME         
        self.object =  Digit(SERIAL_NUMBER, SENSOR_NAME)  
        try:
            self.object.connect()
            self.object.set_fps(FPS_CONFIG)
            self.object.set_intensity(intensity)   
        except:
            print("DIGIT " + SERIAL_NUMBER + " is not connected!")  
            sys.exit(1)
   
        self.current_frame_bgr = []
        self.current_frame_lab = []
        self.diff_bgr = []
        self.diff_lab = []
        self.contact_center = contact_center()
        self.bridge = CvBridge()

        #wait for the light to fully initialize
        time.sleep(1)
        # collecting a set of images to denoise and average into a baseline
        baseline = []
        start_time = time.time()
        while (time.time() - start_time <  2):
            baseline.append(self.object.get_frame())
        self.baseline_bgr = compute_baseline(baseline)
        self.baseline_lab = cv2.cvtColor(self.sensor["baseline"], cv2.COLOR_BGR2LAB)
        
        #Setting all the publishers
        self.DIGIT_PUBLISHERS()
        
    def DIGIT_Publisher (self,rgb_img_topic:str = "rgb", 
                         lab_img_topic:str = "lab", 
                         diff_rgb_img_topic:str = "diff_rgb",
                         diff_lab_img_topic:str = "diff_lab", 
                         output_img_topic:str = "diff_lab", 
                         contact_center_topic:str = "contact_center",                         
                         queue_size:int = 10):    

        self.rgb_publisher = rospy.Publisher("/" + self.name + "/" + rgb_img_topic, Image, queue_size=queue_size)
        self.lab_publisher = rospy.Publisher("/" + self.name + "/" + lab_img_topic, Image, queue_size=queue_size)
        self.diff_rgb_publisher = rospy.Publisher("/" + self.name + "/" + diff_rgb_img_topic, Image, queue_size=queue_size)
        self.diff_lab_publisher = rospy.Publisher("/" + self.name + "/" + diff_lab_img_topic, Image, queue_size=queue_size)
        self.output_publisher = rospy.Publisher("/" + self.name + "/" + output_img_topic, Image, queue_size=queue_size)
        self.contact_center_publisher = rospy.Publisher("/" + self.name + "/" + contact_center_topic, contact_center, queue_size=queue_size)
        
    def run(self):
        self.current_frame_bgr = cv2.GaussianBlur(self.object.get_frame(),(11,11),5)
        self.current_frame_lab = cv2.cvtColor( self.current_frame_bgr, cv2.COLOR_BGR2LAB) 

        base_B,base_G,base_R = cv2.split(self.baseline_bgr)
        base_l,base_a,base_b = cv2.split(self.baseline_lab)
        
        curr_B,curr_G,curr_R = cv2.split(self.current_frame_bgr)
        curr_l,curr_a,curr_b = cv2.split(self.current_frame_lab)
        
        self.diff_lab, res_lab, output_img_lab = compute_diff(curr_b, base_b)
        self.diff_bgr, res_bgr, output_img_bgr = compute_diff(curr_B, base_B)

        poly, (major_axis, major_axis_end), (minor_axis, minor_axis_end), center = res_lab
        self.contact_center.header = Header()
        self.contact_center.header.stamp = rospy.Time.now()
        self.contact_center.center = center
        self.contact_center.major_axis = major_axis
        self.contact_center.major_axis_end = major_axis_end
        self.contact_center.minor_axis = minor_axis
        self.contact_center.minor_axis_end = minor_axis_end

        
        self.rgb_publisher.publish(self.bridge.cv2_to_imgmsg(self.current_frame_bgr))
        self.lab_publisher.publish(self.bridge.cv2_to_imgmsg(self.current_frame_lab))
        self.diff_rgb_publisher.publish(self.bridge.cv2_to_imgmsg(self.diff_bgr))
        self.diff_lab_publisher.publish(self.bridge.cv2_to_imgmsg(self.diff_lab))
        self.output_publisher.publish(self.bridge.cv2_to_imgmsg(output_img_lab))
        self.contact_center_publisher.publish(self.contact_center)
