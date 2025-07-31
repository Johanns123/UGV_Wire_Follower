# Função para ler o arquivo de pontuações e criar a matriz
def read_scores_file(file_path):
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [float(score) for score in line.strip().split()]
            scores.append(row)
    return scores

# Função para encontrar as coordenadas dos objetos com pontuação de 1000000
def find_objects_coordinates(scores):
    objects_coordinates = []
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] == 1000000.0:
                objects_coordinates.append((i, j))
    return objects_coordinates

# Ler o arquivo de pontuações
file_path = 'pontuacoes.txt'
scores = read_scores_file(file_path)

# Encontrar as coordenadas dos objetos com pontuação de 1000000
objects_coordinates = find_objects_coordinates(scores)

# Coordenadas do centro
x_center = 10
y_center = 10  # dado pelo mapa
centro = [x_center, y_center]

# Ajustar as coordenadas dos objetos em relação ao centro
relative_coordinates = [(coord[0] - centro[0], coord[1] - centro[1]) for coord in objects_coordinates]

# Exibir as coordenadas dos objetos ajustadas
print("Coordenadas dos objetos com pontuação de 1000000 (relativas ao centro):")
for coord in relative_coordinates:
    print(coord)
