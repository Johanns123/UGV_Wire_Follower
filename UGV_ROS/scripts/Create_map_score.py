import numpy as np
import xml.etree.ElementTree as ET
import os

def read_world_file(world_file):
    """
    Lê o arquivo .world e extrai as coordenadas dos objetos.
    
    Args:
        world_file (str): Caminho para o arquivo .world.
    
    Returns:
        list: Lista de coordenadas (x, y) dos objetos.
    """
    objects = set()  # Usamos um conjunto para evitar duplicatas
    tree = ET.parse(world_file)
    root = tree.getroot()
    
    for model in root.findall('.//model'):
        pose = model.find('.//pose')
        if pose is not None:
            x, y, *_ = map(float, pose.text.split())
            objects.add((int(round(x)), int(round(y))))  # Arredonda e converte para inteiro
    
    return list(objects)  # Converte o conjunto de volta para uma lista

def calculate_scores(map_size, objects, object_score=1000000, nearby_score=40, default_score=10):
    """
    Calcula a matriz de pontuações com base nas coordenadas dos objetos.
    
    Args:
        map_size (int): Tamanho do mapa (map_size x map_size).
        objects (list): Lista de coordenadas (x, y) dos objetos.
        object_score (int): Pontuação atribuída à posição do objeto.
        nearby_score (int): Pontuação atribuída às posições adjacentes aos objetos.
        default_score (int): Pontuação padrão atribuída a todas as outras posições.
    
    Returns:
        np.ndarray: Matriz de pontuações.
    """
    scores = np.full((map_size, map_size), default_score)  # Inicializa a matriz com a pontuação padrão
    
    center_offset = map_size // 2  # Calcula o deslocamento para centralizar (0,0)

    for (x, y) in objects:
        map_x = x + center_offset  # Ajusta as coordenadas para o índice da matriz
        map_y = center_offset - y  # Inverte o eixo y para que o positivo fique para cima
        
        if 0 <= map_x < map_size and 0 <= map_y < map_size:
            scores[map_y, map_x] = object_score  # Define a pontuação para a posição do objeto
            
            # Define pontuações para as coordenadas próximas do objeto
            for i in range(max(0, map_y-1), min(map_size, map_y+2)):
                for j in range(max(0, map_x-1), min(map_size, map_x+2)):
                    if scores[i, j] == default_score and (i, j) != (map_y, map_x):
                        scores[i, j] = nearby_score  # Define a pontuação para as posições adjacentes
    
    return scores

def save_scores_to_file(scores, output_file):
    """
    Salva a matriz de pontuações em um arquivo .txt.
    
    Args:
        scores (np.ndarray): Matriz de pontuações.
        output_file (str): Caminho para o arquivo de saída.
    """
    with open(output_file, 'w') as file:
        for row in scores:
            file.write(' '.join(map(str, row)) + '\n')

def determine_map_size(objects):
    """
    Determina o tamanho necessário do mapa para incluir todos os objetos, centralizando (0,0).
    
    Args:
        objects (list): Lista de coordenadas (x, y) dos objetos.
    
    Returns:
        int: Tamanho do mapa (dimensão da matriz quadrada).
    """
    if not objects:
        return 1  # Caso não haja objetos, retorna o tamanho mínimo

    min_x = min([x for x, y in objects])
    max_x = max([x for x, y in objects])
    min_y = min([y for x, y in objects])
    max_y = max([y for x, y in objects])
    
    # Calcula a maior distância para determinar o tamanho do mapa
    map_size = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)) * 2 + 1
    
    return map_size

# Caminho do arquivo .world
world_file = '../cpr_gazebo/cpr_agriculture_gazebo/worlds/Obstacles2.world'  # Altere para o caminho do seu arquivo .world
output_file = 'pontuacoes.txt'

# Verifica se o arquivo .world existe
if not os.path.exists(world_file):
    print(f"Erro: O arquivo {world_file} não existe.")
else:
    # Processamento
    objects = read_world_file(world_file)
    
    # Imprime as coordenadas dos objetos
    print("Coordenadas dos objetos:", objects)
    
    # Determina o tamanho necessário do mapa
    map_size = determine_map_size(objects)
    
    # Calcula as pontuações
    scores = calculate_scores(map_size, objects)
    
    # Salva as pontuações em um arquivo .txt
    save_scores_to_file(scores, output_file)
    print(f"Arquivo de pontuações criado: {output_file}")
