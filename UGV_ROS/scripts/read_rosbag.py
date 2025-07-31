#!/usr/bin/env python
import rosbag
import matplotlib.pyplot as plt

# Caminho para o arquivo .bag
bag_path = "2025-07-19-16-57-52.bag"

# Listas para armazenar posições do /odom
odom_xs = []
odom_ys = []

print("🔍 Lendo odometria do rosbag...")

# Abrir o rosbag
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/odom']):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        odom_xs.append(x)
        odom_ys.append(y)

print(f"✅ Total de pontos de odometria: {len(odom_xs)}")

# Caminhos fornecidos manualmente
caminho1 = [(0, 0), (0, 3), (1, 3), (1, 4), (2, 4), (2, 5), (3, 5), (3, 6),
            (4, 6), (4, 7), (5, 7), (5, 8), (6, 8), (6, 9), (7, 9), (7, 10),
            (8, 10), (8, 11), (9, 11), (9, 12), (10, 12), (10,14)]
# Separar coordenadas
c1_xs, c1_ys = zip(*caminho1)

# --- PLOT ---
plt.figure(figsize=(10, 10))

# Odometria
plt.plot(odom_xs, odom_ys, 'b-', label="Robot Track (/odom)", linewidth=2)
plt.scatter(odom_xs[0], odom_ys[0], color='green', label='Begin (odom)', zorder=5)
plt.scatter(odom_xs[-1], odom_ys[-1], color='red', label='End (odom)', zorder=5)

# Caminho 1
plt.plot(c1_xs, c1_ys, 'orange', linestyle='--', linewidth=2, label='Ground Truth')

# Ajustes finais
plt.title("Real Track X Ground Truth")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
