#!/usr/bin/env python3

from models import *
from utils.grasp_stability import *
from utils.contact_area_functions import *
from utils.sensor_functions import *
from digit_ros.srv import inference_request

import rospy

def handle_inference(req):
    grasp_stability = GraspStability("D30030", "D20200", True, "/weights/path")
    inference_result = grasp_stability.run()
    return inference_result

def inference_server():
    rospy.init_node('StabilityPrediction')
    s = rospy.Service('add_two_ints', inference_request, handle_inference)
    print("Ready to perform Inference.")
    rospy.spin()

if __name__ == "__main__":
    inference_server()