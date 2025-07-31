import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# Caminho para seu arquivo .sdf/.world
sdf_path = "../cpr_gazebo/cpr_agriculture_gazebo/worlds/fazenda_wiredV3.world"

# Lê o arquivo SDF
tree = ET.parse(sdf_path)
root = tree.getroot()

# Remove namespace
def strip_namespace(tag):
    return tag.split('}')[-1]

print("🔍 Obstáculos encontrados:")
obstaculos = []

# Verifica <include>
for include in root.iter():
    if strip_namespace(include.tag) == "include":
        pose_elem = include.find("pose")
        uri_elem = include.find("uri")
        if uri_elem is not None and pose_elem is not None:
            pose_values = list(map(float, pose_elem.text.strip().split()))
            uri_text = uri_elem.text.strip()
            model_name = uri_text.split("/")[-1]
            x, y, z = pose_values[:3]
            obstaculos.append((x, y))
            print(f"🪵 {model_name}: posição (x={x:.2f}, y={y:.2f}, z={z:.2f})")

# Verifica <model>
for model in root.iter():
    if strip_namespace(model.tag) == "model":
        pose_elem = model.find("pose")
        if pose_elem is not None:
            pose_values = list(map(float, pose_elem.text.strip().split()))
            model_name = model.attrib.get("name", "sem_nome")
            x, y, z = pose_values[:3]
            obstaculos.append((x, y))
            print(f"🧱 {model_name}: posição (x={x:.2f}, y={y:.2f}, z={z:.2f})")

# Verificação
if not obstaculos:
    print("⚠️ Nenhum obstáculo encontrado.")
    exit()

# Extrai coordenadas
xs, ys = zip(*obstaculos)

# PLOT
plt.figure(figsize=(10, 10))
plt.scatter(xs, ys, c='red', label='Obstacles')
plt.title('Map with Obstacles')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# CRIAÇÃO DO MAPA DE CUSTO
resolucao = 0.25
valor_obstaculo = 1000
alcance = 2
margin = 1

min_x = int(min(xs)) - margin
max_x = int(max(xs)) + margin
min_y = int(min(ys)) - margin
max_y = int(max(ys)) + margin

nx = int((max_x - min_x) / resolucao)
ny = int((max_y - min_y) / resolucao)
mapa = np.zeros((ny, nx), dtype=int)  # << agora é inteiro

# Preenchimento
for ox, oy in obstaculos:
    cx = int((ox - min_x) / resolucao)
    cy = int((oy - min_y) / resolucao)

    raio_celulas = int(alcance / resolucao)
    for dy in range(-raio_celulas, raio_celulas + 1):
        for dx in range(-raio_celulas, raio_celulas + 1):
            x = cx + dx
            y = cy + dy
            if 0 <= x < nx and 0 <= y < ny:
                dist = np.hypot(dx * resolucao, dy * resolucao)
                if dist <= alcance:
                    score = int(round(valor_obstaculo * np.exp(-dist**2 / (2 * (alcance / 2)**2))))
                    mapa[y, x] = max(mapa[y, x], score)

# Salvar como inteiros
np.savetxt("mapa_potencial.txt", mapa, fmt="%d")
print("✅ Mapa salvo como mapa_potencial.txt (valores inteiros)")

# Visualização
plt.imshow(mapa, origin='lower', extent=[min_x, max_x, min_y, max_y], cmap='hot')
plt.colorbar(label="Cost")
plt.title("Potential Map (Cost)")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid(False)
plt.show()
