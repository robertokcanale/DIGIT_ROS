#!/usr/bin/env python3

import rospy
import numpy as np
from utils.contact_area_functions import *
from utils.sensor_functions import *
from digit_ros.src.Digit import DIGIT



def main():
    ns = rospy.get_name() + "/"

    sn1 = rospy.get_param(ns + "sn1")
    name1 = rospy.get_param(ns + "name1")
    rgb_img1 = rospy.get_param(ns + "rgb_img1")
    lab_img1 = rospy.get_param(ns + "lab_img1")
    diff_rgb_img1 = rospy.get_param(ns + "diff_rgb_img1")
    diff_lab_img1 = rospy.get_param(ns + "diff_lab_img1")
    output_img1 = rospy.get_param(ns + "output_img1")
    
    sn2 = rospy.get_param(ns + "sn2")
    name2 = rospy.get_param(ns + "name2")
    rgb_img2 = rospy.get_param(ns + "rgb_img2")
    lab_img2 = rospy.get_param(ns + "lab_img2")
    diff_rgb_img2 = rospy.get_param(ns + "diff_rgb_img2")
    diff_lab_img2 = rospy.get_param(ns + "diff_lab_img2")
    output_img2 = rospy.get_param(ns + "output_img2")
    
    rospy.init_node("digit_ros_" + sn1 + sn2)
    digit1 = DIGIT(sn1, name1)
    digit1.DIGIT_Publisher(rgb_img1, lab_img1, diff_rgb_img1, diff_lab_img1, output_img1)
    digit1.run()
    
    digit2 = DIGIT(sn2, name2)
    digit2.DIGIT_Publisher(rgb_img2, lab_img2, diff_rgb_img2, diff_lab_img2, output_img2)
    digit2.run()
    rospy.spin()

        
if __name__ == '__main__':
    main()