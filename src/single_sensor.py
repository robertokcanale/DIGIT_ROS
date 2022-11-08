import rospy
import numpy as np
from utils.contact_area_functions import *
from utils.sensor_functions import *
from DIGIT import DIGIT



def main():
    ns = rospy.get_name() + "/"

    sn = rospy.get_param(ns + "sn")
    name = rospy.get_param(ns + "name")
    rgb_img = rospy.get_param(ns + "rgb_img")
    lab_img = rospy.get_param(ns + "lab_img")
    diff_rgb_img = rospy.get_param(ns + "diff_rgb_img")
    diff_lab_img = rospy.get_param(ns + "diff_lab_img")
    output_img = rospy.get_param(ns + "output_img")
    
    rospy.init_node("digit_ros_" + sn)
    digit = DIGIT(sn, name)
    digit.DIGIT_Publisher(rgb_img, lab_img, diff_rgb_img, diff_lab_img, output_img)
    digit.run()
    
    rospy.spin()

        
if __name__ == '__main__':
    main()