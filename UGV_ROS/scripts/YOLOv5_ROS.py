#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np

# Inicializa o node ROS
rospy.init_node('yolov5_detector')

# Caminho para o modelo customizado treinado
best_path0 = '/home/johann/CNN_Model/yolov5/runs/train/yolov5m_dataset4_30epochs_wsl/weights/best.pt'

# Carrega o modelo YOLOv5 customizado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=best_path0, source='local')
model.eval()  # Coloca o modelo em modo de avaliação (opcional, dependendo do setup do YOLOv5)
model.verbose = False 

# Cria um objeto CvBridge para converter mensagens ROS para imagens OpenCV
bridge = CvBridge()

# Configuração do tópico de publicação para imagens com detecções YOLO
yolo_image_topic = "/yolo/image_raw"
yolo_image_pub = rospy.Publisher(yolo_image_topic, Image, queue_size=10)

def image_callback(msg):
    # Converte a mensagem de imagem ROS para um formato OpenCV
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # Executa a detecção usando YOLOv5
    results = model(frame)
    
    # Verifique se o objeto 'results' contém as detecções corretas
    if hasattr(results, 'pred'):
        # Acessa as detecções na propriedade 'pred'
        detections = results.pred[0]  # 'pred' é uma lista de tensores para cada imagem processada

        for *xyxy, conf, cls in detections:
            # Converte os valores dos tensores para inteiros
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(conf)
            cls = int(cls)
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Desenha a caixa delimitadora e o label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        rospy.logwarn("O objeto 'results' não possui a propriedade 'pred' esperada.")
    
    # Mostra a imagem com as detecções
    cv2.imshow('YOLOv5 Detection', bridge.imgmsg_to_cv2(msg, "bgr8"))
    cv2.waitKey(1)

    # Converte a imagem processada de volta para uma mensagem ROS e publica
    processed_image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
    yolo_image_pub.publish(processed_image_msg)


# Assina o tópico de imagem da câmera
image_topic = "/camera/color/image_raw"  # Altere para o tópico da sua câmera
rospy.Subscriber(image_topic, Image, image_callback)

# Mantém o node ativo
rospy.spin()
