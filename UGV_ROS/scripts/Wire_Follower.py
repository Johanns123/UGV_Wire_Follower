#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
import torch
import cv2
import warnings
import math
import heapq
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Constantes
LIMIT_ANGULAR_ROBOT_SPEED = 1.5
LIMIT_LINEAR_ROBOT_SPEED = 0.6
OFFSET_FACTOR = 0.25  # Fator de offset ajustável (proporcional à largura da imagem)

# Caminho para o modelo customizado treinado
best_path0 = '/home/johann/CNN_Model/yolov5/runs/train/yolov5m_dataset4_30epochs_wsl/weights/best.pt'

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.position_pub = rospy.Publisher('/robot_position', Odometry, queue_size=10)
        self.bridge = CvBridge()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=best_path0, source='local')
        self.model.eval()
        self.model.verbose = False
        self.wire_info = None
        self.wire_vertices = {}
        self.rate = rospy.Rate(10)  # 10 Hz
        self.current_position = None

    def odom_callback(self, msg):
        # Extraindo informações de posição da mensagem Odometry
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        
        # Atualiza a posição atual
        self.current_position = (pos_x, pos_y, yaw)
        
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        height, width, _ = frame.shape  # Obtém a largura da imagem
        results = self.model(frame)

        if hasattr(results, 'pred') and len(results.pred[0]) > 0:
            detections = results.pred[0]

            for *xyxy, conf, cls in detections:
                if int(cls) == 0:  # Classe 'wire' é a classe 0
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(conf)
                    
                    # Calcula a largura da bounding box
                    bbox_width = x2 - x1
                    
                    # Calcula o ponto central da bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Armazena as informações do fio
                    self.wire_info = {
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": bbox_width,
                        "vertices": (x1, y1, x2, y2)
                    }
                    
                    # Encontra o vértice mais próximo do centro da imagem
                    left_distance = abs(x1 - center_x)
                    right_distance = abs(x2 - center_x)
                    
                    if left_distance < right_distance:
                        closest_vertex_x = x1
                    else:
                        closest_vertex_x = x2
                    
                    self.wire_vertices = {
                        "closest_vertex_x": closest_vertex_x,
                        "vertices": (x1, y1, x2, y2),
                        "image_width": width  # Adiciona a largura da imagem para referência
                    }
                    
                    # Desenha a caixa delimitadora e o label
                    label = f"wire {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        else:
            # Se não encontrar o fio, limpa as informações e faz o robô girar
            self.wire_info = None
            self.wire_vertices = {}
        
        cv2.imshow('YOLOv5 Detection', frame)
        cv2.waitKey(1)

    def move_robot(self):
        self.publish_current_position()
        twist = Twist()
        error = 0
        offset = 0
        pixel_central_da_imagem = 0
        angular_velocity = 0
        center_x = 0
        
        k_p = 0.008  # Ganho proporcional, ajuste conforme necessário
        k_i = 0.00008 #Ganho integrativo
        P = 0
        I = 0
        setpoint =0
        PV = 0
        
        if self.wire_info and self.wire_vertices:
            # closest_vertex_x = self.wire_vertices["closest_vertex_x"]
            center_x = self.wire_info["center_x"]
            image_width = self.wire_vertices["image_width"]
            
            # Calcula o centro da imagem
            pixel_central_da_imagem = image_width // 2

            # Define o offset com base na posição relativa do fio na imagem
            if center_x < pixel_central_da_imagem:
                # Se o fio estiver à esquerda, ajusta o setpoint para a esquerda
                offset = OFFSET_FACTOR * image_width
            else:
                # Se o fio estiver à direita, ajusta o setpoint para a direita
                offset = -OFFSET_FACTOR * image_width
            
            # Calcula o erro como (center_x - OFFSET) - pixel_central_da_imagem
            setpoint = pixel_central_da_imagem - offset
            PV = center_x
            error = setpoint - PV 
            ##PV = pixel_central_da_imagem - offset
            ##setpoint = center_x
            P = error * k_p
            I = (error + error) * k_i
            
            angular_velocity = P + I
            linear_velocity = 0.5  # Velocidade linear constante
            
            twist.angular.z = angular_velocity
            twist.linear.x = linear_velocity
        else:
            # Se não encontrar o fio, gira em torno do próprio eixo
            twist.angular.z = 0.2
            twist.linear.x = 0
        
        # rospy.loginfo(f"erro {error}, setpoint {setpoint}, PV {PV} ang_vel {angular_velocity}")
        self.velocity_pub.publish(twist)

    def publish_current_position(self):
        if self.current_position is not None:
            current_x, current_y, phi = self.current_position
            rospy.loginfo(f'Posição atual: x={current_x}, y={current_y}, phi={phi}')

            # Cria uma mensagem Odometry para publicar a posição atual
            current_position_msg = Odometry()
            current_position_msg.pose.pose.position.x = current_x
            current_position_msg.pose.pose.position.y = current_y
            current_position_msg.pose.pose.position.z = 0.0
            current_position_msg.pose.pose.orientation.w = phi

            # Publica a mensagem no tópico /robot_position
            self.position_pub.publish(current_position_msg)
        else:
            rospy.logwarn("Posição do robô ainda não está disponível.")

            
    def run(self):
        rospy.loginfo('Nó do controlador do robô iniciado.')
        
        while not rospy.is_shutdown():
            self.move_robot()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotController()
        image_topic = "/camera/color/image_raw"  # Altere para o tópico da sua câmera
        rospy.Subscriber(image_topic, Image, controller.image_callback)
        controller.run()

    except rospy.ROSInterruptException:
        pass
