#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
from yolov5 import YOLOv5
from pathlib import Path

class YOLOv5Node:
    def __init__(self):
         # Verify if CUDA is available and set device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        # Load YOLOv5 model
        weights_path = Path('/home/johann/CNN_Model/yolov5/runs/train/yolov5m_dataset4_30epochs_wsl/weights/best.pt')
        self.model = YOLOv5(weights_path, device=self.device)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Modelo YOLOv5
        # self.model = torch.load(Path('/caminho/para/seu/modelo/treinado.pt'), map_location=torch.device('cpu')).autoshape()  # or map_location=torch.device('cuda') if using GPU
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.result_pub = rospy.Publisher("/yolov5/results", String, queue_size=10)
        rospy.loginfo("YOLOv5 node initialized.")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert ROS Image message to CV image: {e}")
            return

        # Add logging to verify image shape and type
        rospy.loginfo(f"Received image with shape: {cv_image.shape} and dtype: {cv_image.dtype}")

        # Convert the image to RGB
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Ensure the image is a tensor and move it to the device
        img_tensor = torch.from_numpy(cv_image_rgb).to(self.device).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW format

        # Perform inference
        results = self.model(img_tensor)

        # Process results
        results_str = results.pandas().xyxy[0].to_json(orient="records")
        rospy.loginfo(f"YOLOv5 results: {results_str}")

        self.result_pub.publish(results_str)

if __name__ == '__main__':
    rospy.init_node('yolov5_node', anonymous=True)
    YOLOv5Node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down YOLOv5 node.")