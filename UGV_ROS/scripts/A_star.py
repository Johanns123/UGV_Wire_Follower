import heapq
import numpy as np
import matplotlib.pyplot as plt

def ler_mapa(arquivo):
    mapa = []
    with open(arquivo, 'r') as f:
        linhas = f.readlines()
        for linha in linhas:  # Não inverter a ordem das linhas
            mapa.append(list(map(int, linha.split())))
    return mapa

def heuristica(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

mapa = ler_mapa("pontuacoes.txt")

linha = len(mapa)
coluna = len(mapa[0])

x_center = coluna// 2
y_center =  linha // 2

centro = [x_center, y_center]

print(centro)

# Definindo os pontos de início e objetivo (na ordem convencional)
inicio = [0, 0]  # Coordenadas centrais
objetivo = [8, -4]  # Coordenadas centrais

x_mapa_inicio = centro[0] + inicio[0]
y_mapa_inicio = centro[1] - inicio[1]
x_mapa_objetivo = centro[0] + objetivo[0]
y_mapa_objetivo = centro[1] - objetivo[1]
inicio_mapa = (x_mapa_inicio, y_mapa_inicio) 
objetivo_mapa = (x_mapa_objetivo , y_mapa_objetivo)
print(inicio_mapa)
print(objetivo_mapa)

# # Executando o algoritmo A*
caminho, custo_total = a_estrela(mapa, inicio_mapa, objetivo_mapa)

if caminho is None:
    print("Nenhum caminho encontrado.")
else:
    print(f"Caminho encontrado: {caminho}")
    print(f"Custo total: {custo_total}")


print(caminho)
# Imprimindo a pontuação de cada coordenada do caminho
print("\nPontuações das coordenadas do caminho encontrado:")
for pos in caminho:
    x, y = pos
    print(f"Coordenada: ({x}, {y}), Pontuação: {mapa[y][x]}")

# # Imprimindo todas as coordenadas e seus custos
# print("\nTodas as coordenadas e seus respectivos custos:")
# for y in range(len(mapa)):
#     for x in range(len(mapa[0])):
#         print(f"Coordenada: ({x}, {y}), Pontuação: {mapa[y][x]}")



# Plotando o caminho encontrado
x_caminho = [pos[0] for pos in caminho]
y_caminho = [pos[1] for pos in caminho]

# plt.plot(x_caminho, y_caminho, marker='o', color='blue')  # Coluna como x e linha como y
# plt.title('Caminho Encontrado')
# plt.xlabel('Coordenada x (coluna)')
# plt.ylabel('Coordenada y (linha)')
# plt.gca().invert_yaxis()  # Inverter eixo y para alinhar com a orientação do mapa
# plt.grid(True)
# plt.show()

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

# plt.plot(x_caminho_real, y_caminho_real, marker='o', color='red')  # Coluna como x e linha como y
# plt.title('Caminho Encontrado')
# plt.xlabel('Coordenada x (coluna)')
# plt.ylabel('Coordenada y (linha)')
# plt.grid(True)
# plt.show()