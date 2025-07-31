#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math

chegou_x = False
chegou_y = False
delta_x = 0
delta_y = 0

# Constantes
DISTANCE_THRESHOLD = 0.05
ROTATION_THRESHOLD = 0.05  # Um valor de tolerância para a orientação
ORIENTATION_TRESHOLD = 0.1
ANGULAR_SPEED = 0.8       # Velocidade angular para rotação

LINEAR_SPEED = 0.8        # Velocidade linear para movimento para frente
ROTATION_RIGHT = math.pi/2   #Rotação esquerda 90 graus
ROTATION_LEFT = - math.pi/2
FRONT = 0.00
BACK = math.pi




class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=True)
        self.velocity_pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/husky_velocity_controller/odom', Odometry, self.odom_callback)
        self.position_pub = rospy.Publisher('/robot_position', Odometry, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.linear_speed = 0.2  # m/s
        self.kp_lin = 0.4
        self.kp_ang = 0.6
        self.angular_speed = 0.5  # rad/s
        self.current_position = None
        self.target_x = None
        self.target_y = None
        self.already_calc = False
        self.arrived_at_target = False  # Flag to track if arrived at target
        self.phi_desired = 0.0
        self.distance_to_goal = 0.0
        self.distance_destination = 0
        self.previous_x = 0
        self.previous_y = 0

    def odom_callback(self, msg):
        # Extracting position information from Odometry message
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        # Update current position
        self.current_position = (pos_x, pos_y, yaw)


    def move_robot(self):
        global chegou_x, chegou_y
        if self.current_position is not None and self.target_x is not None and self.target_y is not None:
            current_x, current_y, phi = self.current_position
            # global x_arrived, y_arrived
            u_x = self.target_x - current_x
            u_y = self.target_y - current_y

            # phi_desired = math.atan2(u_y, u_x)

            # angular_error = math.atan2(math.sin(phi_desired - phi), math.cos(phi_desired - phi))  # Simplified angular error calculation
            # distance_tolerance = 0.3
            # angle_tolerance = 0.2
            # Calcula a distância até o destino
            # distance = ((destination_x - current_x) ** 2 + (destination_y - current_y) ** 2) ** 0.5
            
            phi = math.atan2(math.sin(phi), math.cos(phi))
            rospy.loginfo(f'u_x={u_x}, u_y={u_y}')

            # if not self.already_calc:
            self.phi_desired = math.atan2(u_y, u_x) 
            self.distance_to_goal = math.sqrt((self.target_y**2) + (self.target_x**2))
            self.already_calc = True
                # if u_x < 0 and u_y > 0:
                    # self.phi_desired = math.pi + self.phi_desired
            delta_x = current_x - self.previous_x
            delta_y = current_y - self.previous_y
            distance_error = abs(self.distance_to_goal - math.sqrt((current_x**2) + (current_y**2)))
            angular_error = math.atan2(math.sin(self.phi_desired - phi), math.cos(self.phi_desired - phi))
            rospy.loginfo(f'distance_target={self.distance_to_goal}, delta_dist={math.sqrt((current_x**2) + (current_y**2))}, phi={phi}, phi_desired={self.phi_desired}, distance error={distance_error}')

            self.previous_x = current_x
            self.previous_y = current_y

            if not self.arrived_at_target:
                # if abs(u_x) <= 0.1 and abs(u_y) <= 0.1:
                if abs(distance_error) < 0.06:
                    twist = Twist()  # Stop the robot
                    self.velocity_pub.publish(twist)
                    rospy.loginfo('Robot reached the target position.')
                    # Publish current position when target is reached
                    self.publish_current_position()
                    self.arrived_at_target = True  # Set flag to True
                    self.already_calc = False
                    # Ask user for new target if desired
                    self.ask_for_new_target()

            # Continue moving or rotating until precise position is reached
            if not self.arrived_at_target:
                linear_vel = self.kp_lin * distance_error
                angular_vel = self.kp_ang * angular_error

                twist = Twist()
                twist.linear.x = linear_vel
                twist.angular.z = angular_vel
                self.velocity_pub.publish(twist)

            # angular_error = self.phi_desired - phi

            # Primeiro mova no eixo X
            # if not chegou_x:
            #     if u_x > DISTANCE_THRESHOLD:
            #         if (self.target_x > current_x):
            #             if abs(phi - FRONT) > ROTATION_THRESHOLD:
            #                 if phi > 0:
            #                     rospy.loginfo(f"current rotation, girando direita {phi}")
            #                     self.movement(ANGULAR_SPEED, LINEAR_SPEED, "right")
            #                 elif phi < 0:
            #                     rospy.loginfo(f"current rotation, girando esquerda {phi}")
            #                     self.movement(ANGULAR_SPEED, LINEAR_SPEED, "left")
                                
            #             else:
            #                 rospy.loginfo("vá para a frente")
            #                 self.movement(ANGULAR_SPEED, LINEAR_SPEED, "front")
                            
            #         elif (self.target_x < current_x):
            #             if abs(abs(phi) - BACK) > ROTATION_THRESHOLD:
            #                 if phi > 0:
            #                     rospy.loginfo(f"current rotation, girando esquerda {phi}")
            #                     self.movement(ANGULAR_SPEED, LINEAR_SPEED, "left")
                                
            #                 elif phi < 0:
            #                     rospy.loginfo(f"current rotation, girando direita {phi}")
            #                     self.movement(ANGULAR_SPEED, LINEAR_SPEED, "right")
                                
            #             else:
            #                 rospy.loginfo("vá para a frente")
            #                 self.movement(ANGULAR_SPEED, LINEAR_SPEED, "front")
                    
            #     else:
            #         chegou_x = True
            
            # elif not chegou_y:
            #     if u_y > DISTANCE_THRESHOLD:
            #         if (self.target_y > current_y):
            #             if abs(phi - ROTATION_RIGHT) > ROTATION_THRESHOLD:
            #                 if (phi > ROTATION_RIGHT):
            #                     rospy.loginfo("movendo no y")
            #                     self.movement(ANGULAR_SPEED, LINEAR_SPEED, "right")
            #                       # Ou "left", dependendo da orientação desejada
            #                 elif phi < ROTATION_RIGHT:
            #                     rospy.loginfo("movendo no y")
            #                     self.movement(ANGULAR_SPEED, LINEAR_SPEED, "left")
            #                       # Ou "left", dependendo da orientação desejada
            #             else:
            #                 self.movement(ANGULAR_SPEED, LINEAR_SPEED, "front")
            #                 rospy.loginfo("andando frente no y")
                            
            #         elif (self.target_y < current_y):
            #             if (abs(ROTATION_LEFT - phi) > ROTATION_THRESHOLD):
            #                 if abs(phi) > abs(ROTATION_LEFT):
            #                     self.movement(ANGULAR_SPEED, LINEAR_SPEED, "left")
            #                     rospy.loginfo("movendo no y")
            #                       # Ou "left", dependendo da orientação desejada
            #                 elif abs(phi) < abs(ROTATION_LEFT):
            #                     self.movement(ANGULAR_SPEED, LINEAR_SPEED, "right")
            #                     rospy.loginfo("movendo no y")
            #                       # Ou "left", dependendo da orientação desejada
            #             else:
            #                 rospy.loginfo("andando frente no y")
            #                 self.movement(ANGULAR_SPEED, LINEAR_SPEED, "front")
                            
            #     else:
            #         chegou_y = True

            # if chegou_x and chegou_y:
            #     # Chegou ao destino, avança para o próximo
            #     twist = Twist()  # Stop the robot
            #     self.velocity_pub.publish(twist)
            #     rospy.loginfo(f'Robot reached the target position.')
            #     # Publish current position when target is reached
            #     self.publish_current_position()
            #     self.arrived_at_target = True  # Set flag to True
            #     chegou_x = False
            #     chegou_y = False
            #     # Ask user for new target if desired
            #     self.ask_for_new_target()

                # next_destination()
                #publish_cmd_vel

            

            # if not self.arrived_at_target:
                
                # if(angular_error) > angle_tolerance:
                    # if u_x > 0 and u_y > 0:
                    #     angular_error = self.phi_desired - phi
                    #     #rotate left
                    #     angular_vel = self.kp_ang * angular_error
                    #     rospy.loginfo("Rotate Left")
                    # elif u_x > 0 and u_y < 0:
                    #     #rotate right
                    #     angular_error = self.phi_desired - phi
                    #     angular_vel = self.kp_ang * (-angular_error)
                    #     rospy.loginfo("Rotate Right")
                    # if u_x < 0 and u_y > 0:
                        # angular_error = (math.pi+self.phi_desired) - phi
                        #rotate left
                        # angular_error = self.phi_desired - phi
                    # angular_vel = self.kp_ang * angular_error
                    # rospy.loginfo("Rotate Left")
                    # elif u_x < 0 and u_y < 0:
                    #     #rotate left
                    #     angular_error = self.phi_desired - phi
                    #     angular_vel = self.kp_ang * angular_error
                    #     rospy.loginfo("Rotate Left")

                    # linear_vel = 0

                    
                
                # else:
                    # if(distance_to_goal) > distance_tolerance:
                    #     linear_vel = self.kp_lin * distance_to_goal
                    #     angular_vel = 0

                    #     twist = Twist()
                    #     twist.linear.x = linear_vel
                    #     twist.angular.z = angular_vel
                    #     self.velocity_pub.publish(twist)

                    # else:
                    #     twist = Twist()  # Stop the robot
                    #     self.velocity_pub.publish(twist)
                    #     rospy.loginfo(f'Robot reached the target position.')
                    #     # Publish current position when target is reached
                    #     self.publish_current_position()
                    #     self.arrived_at_target = True  # Set flag to True
                    #     self.already_calc = False
                    #     # Ask user for new target if desired
                    #     self.ask_for_new_target()

                # if abs(u_x) <= 0.05 and abs(u_y) <= 0.05:
                #     twist = Twist()  # Stop the robot
                #     self.velocity_pub.publish(twist)
                #     rospy.loginfo(f'Robot reached the target position.')
                #     # Publish current position when target is reached
                #     self.publish_current_position()
                #     self.arrived_at_target = True  # Set flag to True

                #     # Ask user for new target if desired
                #     self.ask_for_new_target()
                
                # else:
                #     angular_vel = self.kp_ang * angular_error
                #     linear_vel = self.kp_lin * distance_to_goal
                #     twist = Twist()
                #     twist.linear.x = linear_vel
                #     twist.angular.z = angular_vel
                #     self.velocity_pub.publish(twist)

    def movement(self, angular_speed, linear_speed, direction):

        if direction == "front":
            angular_vel = 0
            linear_vel = self.kp_lin * linear_speed

        angular_vel = self.kp_ang * (-angular_speed) if direction == "right" else self.kp_ang * angular_speed 
        if direction == "right" or direction == "left":
            linear_vel = 0  
        
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.velocity_pub.publish(twist)       

    def publish_current_position(self):
        if self.current_position is not None:
            current_x, current_y, phi = self.current_position
            rospy.loginfo(f'Current position: x={current_x}, y={current_y}, phi={phi}')
            
            # Create an Odometry message to publish the current position
            current_position_msg = Odometry()
            current_position_msg.pose.pose.position.x = current_x
            current_position_msg.pose.pose.position.y = current_y
            current_position_msg.pose.pose.position.z = 0.0  # Assume z=0 for simplicity
            current_position_msg.pose.pose.orientation.w = phi  # Only w is set for simplicity

            # Publish the message to the /robot_position topic
            self.position_pub.publish(current_position_msg)
        elif answer.lower() == 'no':
            rospy.loginfo('Exiting the program.')
            rospy.signal_shutdown('User requested program exit.')
            
    def ask_for_new_target(self):
        answer = input('Do you want to enter a new target point? (yes/no): ')
        if answer.lower() == 'yes':
            self.target_x = float(input('Enter the new target x coordinate: '))
            self.target_y = float(input('Enter the new target y coordinate: '))
            self.arrived_at_target = False  # Reset the flag for the new target
        
    def run(self):
        rospy.loginfo('Robot controller node started.')
        while not rospy.is_shutdown():
            self.move_robot()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotController()
        # Get target point from user input via terminal
        target_x = float(input('Enter the target x coordinate: '))
        target_y = float(input('Enter the target y coordinate: '))
        controller.target_x = target_x
        controller.target_y = target_y
        controller.run()
    except rospy.ROSInterruptException:
        pass
