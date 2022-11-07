from digit_interface import Digit
import cv2, rospy
import numpy as np
from utils.contact_area_functions import *
from utils.sensor_functions import *
from sensor_msgs.msg import Image
from DIGIT_ROS.msg import contact_center

SENSOR_SERIAL_NUMBER_LIST = ["D20492"]
SENSOR_NAME_LIST = ["LeftGripper"]
FPS_CONFIG = Digit.STREAMS["QVGA"]["fps"]["60fps"]
DIGIT_INTENSITY = 10

class DIGIT:
    def __init__ (self, SERIAL_NUMBER, SENSOR_NAME, FPS_CONFIG = Digit.STREAMS["QVGA"]["fps"]["60fps"], intensity:int = 10):
        self.sn = SERIAL_NUMBER         
        self.name = SENSOR_NAME         
        self.object =  Digit(SERIAL_NUMBER, SENSOR_NAME)     
        self.object.connect()
        self.object.set_fps(FPS_CONFIG)
        self.object.set_intensity(intensity)        
        
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
        
    def DIGIT_PUBLISHERS (self,rgb_img:str = "rgb", 
                         lab_img:str = "lab", 
                         diff_rgb_img:str = "diff_rgb",
                         diff_lab_img:str = "diff_lab", 
                         contact_center:str = "contact_center",                         
                         queue_size:int = 10):    

        self.rgb_publisher = rospy.Publisher("/" + self.name + "/" + rgb_img, Image, queue_size=queue_size)
        self.lab_publisher = rospy.Publisher("/" + self.name + "/" + lab_img, Image, queue_size=queue_size)
        self.diff_rgb_publisher = rospy.Publisher("/" + self.name + "/" + diff_rgb_img, Image, queue_size=queue_size)
        self.diff_lab_publisher = rospy.Publisher("/" + self.name + "/" + diff_lab_img, Image, queue_size=queue_size)
        self.contact_center_publisher = rospy.Publisher("/" + self.name + "/" + contact_center, contact_center, queue_size=queue_size)
        
    def read():
        while True:
            