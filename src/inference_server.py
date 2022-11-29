#!/usr/bin/env python3

from __future__ import print_function
from digit_ros.srv import inference_request

import rospy

def handle_ainference(req):

    inference_result = 1
    return inference_result

def inference_server():
    rospy.init_node('StabilityPrediction')
    s = rospy.Service('add_two_ints', inference_request, handle_ainference)
    print("Ready to perform Inference.")
    rospy.spin()

if __name__ == "__main__":
    inference_server()