#!/usr/bin/env python2
from __future__ import print_function

import os
import roslib
roslib.load_manifest('deidentify')
import sys
import rospy
import cv2
import numpy as np
import tensorflow as tf
from time import sleep, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def img_prep(mdl, frame):
    input_shape = tuple(mdl.input_details[0]['shape'][1:3][::-1])
    im = cv2.resize(frame, input_shape)
    im = np.expand_dims(im, axis=0)
    im = im / 255
    return im.astype(mdl.input_details[0]['dtype'])

def post_process(res, frame, kern_size):
    res = np.argmax(res, axis=3)[0]
    res = 255 * (res==15).astype(np.uint8)
    mask = cv2.resize(res, frame.shape[:2][::-1])
    gray = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    not_gray = cv2.bitwise_not(gray)

    blur = cv2.medianBlur((frame).astype(np.uint8), kern_size)

    im1 = cv2.bitwise_and(blur, gray)
    im2 = cv2.bitwise_and(frame, not_gray)

    return cv2.bitwise_or(im1, im2)

class tflite_inference(object):

    def __init__(self):
        return

    def load_model(self, mdl_path):
        self.interpreter = tf.lite.Interpreter(mdl_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        return

    def run_inference(self, im):
        start_time = time()
        self.interpreter.set_tensor(self.input_details[0]['index'], im) 
        self.interpreter.invoke() 
        results = self.output_details[0]['index']
        self.res = self.interpreter.get_tensor(results)
        end_time = time()
        return self.res, end_time - start_time


class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("/images/deidentified", Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)

  def callback(self, data):
    try:
      frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    im = img_prep(mdl, frame.astype(np.float32))
    res, elapsed = mdl.run_inference(im)

    print(elapsed)

    args = (res, frame, kern_size)
    result = post_process(*args)

    cv2.imshow("Image window", result)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(result, "passthrough"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    kern_size = 31
    mdl_file = 'deeplabv3_257_mv_gpu.tflite'
    MDL_DIR = os.environ['HOME'] + '/Downloads/' 
    mdl_path = os.path.join(MDL_DIR, mdl_file)


    mdl = tflite_inference()
    mdl.load_model(mdl_path)

    main(sys.argv)





