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
    def __init__ (self, SERIAL_NUMBER, SENSOR_NAME, FPS_CONFIG = Digit.STREAMS["QVGA"]["fps"]["60fps"], intensity = 10):
        self.sn = SERIAL_NUMBER         
        self.name = SENSOR_NAME
        self.object =  Digit(SERIAL_NUMBER, SENSOR_NAME)  
        try:
            self.object.connect()
            self.object.set_fps(FPS_CONFIG)
            self.object.set_intensity(intensity)   
        except Exception as e:
            print(e)
            print("DIGIT " + SERIAL_NUMBER + " is not connected!")  
            sys.exit(1)
   
        self.current_frame_bgr = Image()
        self.current_frame_lab = Image()
        self.diff_bgr = Image()
        self.diff_lab = Image()
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
        self.baseline_lab = cv2.cvtColor(self.baseline_bgr, cv2.COLOR_BGR2LAB)
        
    def DIGIT_Publishers(self,rgb_img_topic:str = "rgb", 
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
        # self.current_frame_bgr =self.object.get_frame() #cv2.GaussianBlur(self.object.get_frame(),(11,11),5)
        self.current_frame_bgr = cv2.GaussianBlur(self.object.get_frame(),(11,11),5)
        self.current_frame_lab = cv2.cvtColor( self.current_frame_bgr, cv2.COLOR_BGR2LAB) 

        base_B,base_G,base_R = cv2.split(self.baseline_bgr)
        base_l,base_a,base_b = cv2.split(self.baseline_lab)
        
        curr_B,curr_G,curr_R = cv2.split(self.current_frame_bgr)
        curr_l,curr_a,curr_b = cv2.split(self.current_frame_lab)
        
        self.diff_lab, res_lab = contact_area(target=curr_b.copy(),base=base_b.copy())
        self.diff_bgr, res_bgr = contact_area(target=self.current_frame_bgr.copy(),base=self.baseline_bgr.copy())

        if not res_lab is None:
            poly, (major_axis, major_axis_end), (minor_axis, minor_axis_end), center = res_lab
            self.contact_center.header = Header()
            self.contact_center.header.stamp = rospy.Time.now()
            self.contact_center.center.x = center[0]
            self.contact_center.center.y = center[1]
            self.contact_center.major_axis.x  = major_axis[0]
            self.contact_center.major_axis.y = major_axis[1]
            self.contact_center.major_axis_end.x  = major_axis_end[0]
            self.contact_center.major_axis_end.y = major_axis_end[1]
            self.contact_center.minor_axis.x  = minor_axis[0]
            self.contact_center.minor_axis.y = minor_axis[1]
            self.contact_center.minor_axis_end.x  = minor_axis_end[0]
            self.contact_center.minor_axis_end.y = minor_axis_end[1]
            self.contact_center_publisher.publish(self.contact_center)
            
            output_img_lab = draw_major_minor(cv2.cvtColor(curr_b.copy(), cv2.COLOR_GRAY2BGR), poly, major_axis, major_axis_end, minor_axis, minor_axis_end)
        else:
                output_img_lab = curr_b
                output_img_lab = cv2.cvtColor(output_img_lab, cv2.COLOR_GRAY2BGR)

        self.rgb_publisher.publish(self.bridge.cv2_to_imgmsg(self.current_frame_bgr))
        self.lab_publisher.publish(self.bridge.cv2_to_imgmsg(self.current_frame_lab))
        self.diff_rgb_publisher.publish(self.bridge.cv2_to_imgmsg(self.diff_bgr))
        self.diff_lab_publisher.publish(self.bridge.cv2_to_imgmsg(self.diff_lab))
        self.output_publisher.publish(self.bridge.cv2_to_imgmsg(output_img_lab))

