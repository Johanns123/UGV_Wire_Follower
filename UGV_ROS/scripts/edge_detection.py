#! /usr/bin/env python

from __future__ import print_function

import roslib 
import sys 
import rospy 
import cv2 
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
import heapq
import numpy as np


class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("/camera/color/edges", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Converte para escala de cinza
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray,(5,5),0)

        # Realiza a detecção de bordas utilizando Canny
        edges = cv2.Canny(blur, 30, 140)

        # Encontra os contornos na imagem de bordas
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop sobre todos os contornos encontrados
        for contour in contours:
            # Calcula os momentos do contorno
            M = cv2.moments(contour)

            # Verifica se o contorno é válido para evitar divisão por zero
            if M["m00"] != 0:
                # Calcula as coordenadas do centroide
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Desenha o centroide na imagem colorida
                cv2.circle(cv_image, (cX, cY), 5, (0, 255, 0), -1)

        edges_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        cv2.imshow("Edge window", edges_image)
        cv2.waitKey(3)


        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(edges_image, "bgr8"))
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
    main(sys.argv)

