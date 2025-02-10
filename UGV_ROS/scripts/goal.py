#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
import heapq
import numpy as np

# Constantes
LIMIT_ANGULAR_ROBOT_SPEED = 1.5
LIMIT_LINEAR_ROBOT_SPEED = 0.6

class RobotController:
    def __init__(self):
        rospy.init_node('wire_follower')
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.position_pub = rospy.Publisher('/robot_position', Odometry, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.linear_speed = 0.2  # m/s
        self.kp_lin = 0.4
        self.kp_ang = 0.6
        self.angular_speed = 0.5  # rad/s
        self.current_position = None
        self.destinations = [(0.0, 0.0), (2.0, 1.0)]  # Exemplo de lista de destinos
        self.destination_index = 0
        self.arrived_at_target = False  # Flag para rastrear se chegou ao destino
        self.phi_desired = 0.0
        self.distance_to_goal = 0.0
        self.previous_x = 0
        self.previous_y = 0
        self.angular_error_k1 = 0
        self.distance_error_k1 = 0
        self.uk_ang_k1 = 0
        self.uk_disp_k1 = 0
        self.contador = 0
        
    def odom_callback(self, msg):
        # Extraindo informações de posição da mensagem Odometry
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        
        # Atualiza a posição atual
        self.current_position = (pos_x, pos_y, yaw)

    def move_robot(self):
        if self.current_position is not None and self.destination_index < len(self.destinations):
            current_x, current_y, phi = self.current_position
            target_x, target_y = self.destinations[self.destination_index]

            u_x = target_x - current_x
            u_y = target_y - current_y

            phi = math.atan2(math.sin(phi), math.cos(phi))

            self.phi_desired = math.atan2(u_y, u_x) 

            delta_x = current_x - self.previous_x
            delta_y = current_y - self.previous_y

            distance_error = abs(math.sqrt(((target_y - current_y) ** 2) + (target_x - current_x) ** 2))
            angular_error = math.atan2(math.sin(self.phi_desired - phi), math.cos(self.phi_desired - phi))

            self.previous_x = current_x
            self.previous_y = current_y

            uk_ang = 1.4939 * angular_error - 1.442808 * self.angular_error_k1 + self.uk_ang_k1
            uk_disp = 2.8154 * distance_error - 2.719113 * self.distance_error_k1 + self.uk_disp_k1

            if math.fabs(angular_error) > 0.01:
                LIMIT_LINEAR_ROBOT_SPEED = 0.2
            else:
                LIMIT_LINEAR_ROBOT_SPEED = 1.8

            if uk_ang > LIMIT_ANGULAR_ROBOT_SPEED:
                uk_ang = LIMIT_ANGULAR_ROBOT_SPEED

            elif uk_ang < -LIMIT_ANGULAR_ROBOT_SPEED:
                uk_ang = -LIMIT_ANGULAR_ROBOT_SPEED

            if uk_disp > LIMIT_LINEAR_ROBOT_SPEED:
                uk_disp = LIMIT_LINEAR_ROBOT_SPEED

            elif uk_disp < -LIMIT_LINEAR_ROBOT_SPEED:
                uk_disp = -LIMIT_LINEAR_ROBOT_SPEED

            self.angular_error_k1 = angular_error
            self.distance_error_k1 = distance_error
            self.uk_ang_k1 = uk_ang
            self.uk_disp_k1 = uk_disp

            if abs(u_x) <= 0.1 and abs(u_y) <= 0.1:
                twist = Twist()  # Para o robô
                self.velocity_pub.publish(twist)
                rospy.loginfo('Robô chegou à posição de destino.')
                # Publica a posição atual quando o destino é alcançado
                self.publish_current_position()
                self.destination_index += 1  # Passa para o próximo destino

            linear_vel = uk_disp
            angular_vel = uk_ang

            twist = Twist()
            twist.linear.x = linear_vel
            twist.angular.z = angular_vel
            self.velocity_pub.publish(twist)

            if self.contador < 10:
                self.contador+= 1
            else: 
                rospy.loginfo(f'Posição atual: x={current_x}, y={current_y}, phi={phi}')
                self.contador = 1
            
            if self.destination_index >= len(self.destinations):
                rospy.loginfo("Encerrado")
                rospy.signal_shutdown("")

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
            rospy.loginfo('Encerrando o programa.')
            rospy.signal_shutdown('Usuário solicitou a saída do programa.')

    def run(self):
        rospy.loginfo('Nó do controlador do robô iniciado.')
        
        # Aguardar até que a posição atual do robô seja recebida
        while self.current_position is None and not rospy.is_shutdown():
            rospy.loginfo('Aguardando a posição atual do robô...')
            rospy.sleep(1)  # Aguarda 1 segundo antes de verificar novamente
        
        # Uma vez que a posição atual do robô é recebida, o código continuará executando
        while not rospy.is_shutdown():
            self.move_robot()
            self.rate.sleep()

def ler_mapa(arquivo):
    mapa = []
    with open(arquivo, 'r') as f:
        linhas = f.readlines()
        for linha in linhas:  # Não inverter a ordem das linhas
            mapa.append(list(map(int, linha.split())))
    return mapa

def heuristica(a, b):
    # return abs(a[0] - b[0]) + abs(a[1] - b[1])
    return np.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))

def a_estrela(mapa, inicio, objetivo):
    linhas, colunas = len(mapa), len(mapa[0])
    movimento = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    fronteira = []
    heapq.heappush(fronteira, (0, inicio))
    origem = {inicio: None}
    custo_ate_agora = {inicio: 0}
    
    while fronteira:
        _, atual = heapq.heappop(fronteira)
        
        if atual == objetivo:
            break
        
        for dx, dy in movimento:
            vizinho = (atual[0] + dx, atual[1] + dy)
            
            if 0 <= vizinho[1] < colunas and 0 <= vizinho[0] < linhas:
                novo_custo = custo_ate_agora[atual] + mapa[vizinho[1]][vizinho[0]]
                
                if vizinho not in custo_ate_agora or novo_custo < custo_ate_agora[vizinho]:
                    custo_ate_agora[vizinho] = novo_custo
                    prioridade = novo_custo + heuristica(objetivo, vizinho)
                    heapq.heappush(fronteira, (prioridade, vizinho))
                    origem[vizinho] = atual
    
    if objetivo not in origem:
        return None, float('inf')
    
    caminho = []
    atual = objetivo
    while atual != inicio:
        caminho.append(atual)
        atual = origem[atual]
    caminho.append(inicio)
    caminho.reverse()
    
    return caminho, custo_ate_agora[objetivo]

def simplificar_caminho(caminho):
    if not caminho:
        return []
    
    simplificado = [caminho[0]]  # Adiciona o ponto inicial
    direcao_atual = (caminho[1][0] - caminho[0][0], caminho[1][1] - caminho[0][1])
    
    for i in range(1, len(caminho) - 1):
        direcao_proxima = (caminho[i+1][0] - caminho[i][0], caminho[i+1][1] - caminho[i][1])
        if direcao_proxima != direcao_atual:
            simplificado.append(caminho[i])
            direcao_atual = direcao_proxima
    
    simplificado.append(caminho[-1])  # Adiciona o ponto final

    return simplificado

if __name__ == '__main__':

    try:
        controller = RobotController()

        mapa = ler_mapa("pontuacoes.txt")

        linha = len(mapa)
        coluna = len(mapa[0])

        x_center = coluna// 2
        y_center =  linha // 2

        centro = [x_center, y_center]

        print("Coordenadas centrais do mapa txt", centro)

        # Definindo os pontos de início e objetivo (na ordem convencional)
        inicio = [0, 0]  # Coordenadas centrais
        objetivo = [8, -4]  # Coordenadas centrais

        x_mapa_inicio = centro[0] + inicio[0]
        y_mapa_inicio = centro[1] - inicio[1]
        x_mapa_objetivo = centro[0] + objetivo[0]
        y_mapa_objetivo = centro[1] - objetivo[1]
        inicio_mapa = (x_mapa_inicio, y_mapa_inicio)
        # inicio_mapa = (0, 0) 
        objetivo_mapa = (x_mapa_objetivo , y_mapa_objetivo)
        print("Inicio", inicio_mapa)
        print("Destino", objetivo_mapa)

        # # Executando o algoritmo A*
        caminho, custo_total = a_estrela(mapa, inicio_mapa, objetivo_mapa)

        if caminho is None:
            print("Nenhum caminho encontrado.")
        else:
            print(f"Caminho encontrado: {caminho}")
            print(f"Custo total: {custo_total}")

        # Imprimindo a pontuação de cada coordenada do caminho
        print("\nPontuações das coordenadas do caminho encontrado:")
        for pos in caminho:
            x, y = pos
            print(f"Coordenada: ({x}, {y}), Pontuação: {mapa[y][x]}")
        
        x_caminho = [pos[0] for pos in caminho]
        y_caminho = [pos[1] for pos in caminho]
        
        x_caminho_real = []
        y_caminho_real = []

        #caminho real
        for caminho in range(min(len(x_caminho), len(y_caminho))):
            # diff_x = x_caminho[caminho] - inicio[0]
            # diff_y = y_caminho[caminho] - inicio[1]
            diff_x = x_caminho[caminho] - centro[0]
            diff_y = centro[1] - y_caminho[caminho]

            x_caminho_real.append(diff_x)
            
            y_caminho_real.append(diff_y)     
                    
        print(x_caminho)
        print(x_caminho_real)
        print(y_caminho)
        print(y_caminho_real)

        caminho_real = []
        caminho_real = [(x, y) for x, y in zip(x_caminho_real, y_caminho_real)]


        print(caminho_real)
            
        # Simplificando o caminho encontrado
        caminho_simplificado = simplificar_caminho(caminho_real)
        print(f"Caminho simplificado: {caminho_simplificado}")

        controller.destinations = [pos for pos in caminho_simplificado]
        controller.run()

    except rospy.ROSInterruptException:
        pass
