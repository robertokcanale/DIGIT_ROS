#!/usr/bin/env python3

import rospy
import numpy as np
from utils.contact_area_functions import *
from utils.sensor_functions import *
from Digit import DIGIT



def main():
    rospy.init_node("digit_ros")
    ns =  "/"
    sn = rospy.get_param(ns + "sn")
    name = rospy.get_param(ns + "name")
    rgb_img = rospy.get_param(ns + "rgb_img")
    lab_img = rospy.get_param(ns + "lab_img")
    diff_rgb_img = rospy.get_param(ns + "diff_rgb_img")
    diff_lab_img = rospy.get_param(ns + "diff_lab_img")
    output_img = rospy.get_param(ns + "output_img")
    
    digit = DIGIT(sn, name)
    digit.DIGIT_Publishers(rgb_img, lab_img, diff_rgb_img, diff_lab_img, output_img)
    while rospy.is_shutdown:
        digit.run()
    
       
if __name__ == '__main__':
    main()
