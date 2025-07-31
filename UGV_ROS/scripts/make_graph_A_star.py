import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Dados fornecidos (x, y, phi)
coordinates = [
    (-0.00024194229114841062, 0.0004194743699515886),
    (0.11394360124475622, 0.01049361144360825),
    (0.29102662104492805, 0.09804139673061786),
    (0.40352985754799514, 0.2551381530968105),
    (0.42797437244430264, 0.4468396675466004),
    (0.36615239193035326, 0.6398002983768685),
    (0.2905911728858075, 0.8212055077004903),
    (0.23022565108292944, 1.0169460230984197),
    (0.1901803291310522, 1.2095434923643378),
    (0.16199093959754815, 1.4042840378477048),
    (0.14033008619474924, 1.6082009083531963),
    (0.12393390484821973, 1.8043168449765052),
    (0.10964052343923084, 2.006942443879616),
    (0.07829577458105123, 2.514810162092312),
    (0.04493635492059012, 3.0971311854916386),
    (0.025554690752553965, 3.4540704576429975),
    (0.0182039759388942, 3.601353073947353),
    (0.011873356148844234, 3.738803900416649),
    (0.00799809214457121, 3.8250754174319828),
    (0.005481877495460234, 3.8810021958847196),
    (0.004519441549269573, 3.9022338412068724),
    (0.001503952431328386, 3.950494821517013),
    (0.0967667994853423, 4.094697790522308),
    (0.2786668905086486, 4.157582534340472),
    (0.4817667486305147, 4.181363020372178),
    (0.6782587584925146, 4.19159281103626),
    (0.883189338996278, 4.196574451180478),
    (1.079982880280227, 4.198280592636868),
    (1.2934486730066006, 4.198096946115271),
    (1.8323546236779717, 4.194181942535445),
    (2.418889756419894, 4.188260517746181),
    (3.009029951919679, 4.181342676142163),
    (3.6240369852237553, 4.173311890127075),
    (4.214378769740861, 4.1650528906714035),
    (4.829242312433313, 4.156095722314222),
    (5.419576727418699, 4.147290245084244),
    (6.009909351061802, 4.138367773628483),
    (6.624836945434286, 4.129010499370106),
    (7.215168012345975, 4.119988210071412),
    (7.830164438381775, 4.110559349677744),
    (8.420494789277473, 4.101490635326785),
    (9.010824983616377, 4.092411939373882),
    (9.625684612369986, 4.082948972167647),
    (10.216014603908798, 4.073857137445512),
    (10.83094175120373, 4.0643815207161955),
    (11.421271600909172, 4.055280438708435),
    (12.011601398576143, 4.046176080181185),
    (12.626523050958323, 4.036684018487654),
    (13.216852723601091, 4.027571509603908),
    (13.831783917350432, 4.018078791519219),
    (14.422113397303514, 4.008953848332753),
    (14.833913388397688, 4.002579564627169),
    (14.8263625207401, 4.002656607750498),
    (14.888889064032279, 4.00165997771062),
    (14.901784679026674, 4.001453072032025)
]

# Separando as coordenadas em dois vetores distintos
x_coords = [coord[0] for coord in coordinates]
y_coords = [coord[1] for coord in coordinates]

# Ponto de referência
ref_points_x = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
ref_points_y = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  4,  4,  4,  4,  4,  4]

# Coordenadas dos quadrados
square_coords_x = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8,8,8,8,8,10,10,10,10,10,12,12,12,12,12,14,14,14,14,14]
square_coords_y = [0, 1, 2, 3, 3.5, 0, 1, 2, 3,3.5, 0, 1, 2, 3, 3.5,0,1,2,3,3.5,0,1,2,3,3.5,0,1,2,3,3.5,0,1,2,3,3.5]

# Criando o gráfico
plt.figure(figsize=(10, 6))
plt.title("Real track X Ideal Path", fontsize=18)
plt.plot(x_coords, y_coords, label='Real Track', marker='o', markersize=6, linestyle='-')
plt.plot(ref_points_x, ref_points_y, color='red', label='Reference Track', linewidth=3)
plt.scatter(square_coords_x, square_coords_y, color='green', label='Solar Panels', marker='s', s=200)  # Adiciona quadrados
plt.scatter(15, 4, color='orange', marker='X', label = "Drone", s=500)  # 's' indica quadrado, 's' para size (tamanho do marcador)
plt.xlabel('X[m]', fontsize=18)
plt.ylabel('Y[m]', fontsize=18)
plt.legend(fontsize=18)
# Aumentando o tamanho das fontes dos dados no eixo X e Y
plt.gca().tick_params(axis='x', labelsize=18)  # Eixo X
plt.gca().tick_params(axis='y', labelsize=18)  # Eixo Y
plt.grid()
plt.show()


# Caminho esperado
path_expected = [(0, y) for y in range(5)] + [(x, 4) for x in range(1, 16)]
path_measured = coordinates

# Calcular desvios
total_deviation = 0
total_path_length = 0
total_real_length = 0

for measured, expected in zip(path_measured, path_expected):
    deviation = euclidean_distance(measured, expected)
    total_deviation += deviation
    
    # print(deviation, total_deviation)

# Comprimento total do caminho esperado
for i in range(1, len(path_expected)):
    total_path_length += euclidean_distance(path_expected[i-1], path_expected[i])

for i in range(1, len(path_measured)):
    total_real_length += euclidean_distance(path_measured[i-1], path_measured[i])
    
# Percentual de desvio
percentual_desvio = ((total_real_length - total_path_length) / total_path_length) * 100

print(f"O percentual de desvio é: {percentual_desvio:.2f}%")


# import matplotlib.pyplot as plt

# # Dados das coordenadas x e y
# x_coords = [
#     -0.004280377143851366, 0.012733055732324782, 0.10554559209174566, 0.2748626673642242,
#     0.46772571935583357, 0.6432483845749819, 0.7625482985152577, 0.7876125727744728,
#     0.708298924195861, 0.5819574136071786, 0.46124119417300513, 0.3594231744117742,
#     0.28136041721228794, 0.21471716158520393, 0.1593428926860345, 0.1097115613974731,
#     0.06999521524778667, 0.0476400711339603, 0.031878875729858716, 0.02320169086247761,
#     0.020594938304038257, 0.04028346674147586, 0.201320653622369, 0.39204183841893886,
#     0.596076443437042, 0.7929722028659842, 0.9897908410488332, 1.1947500977196446,
#     1.3946342303556167, 1.8761166791488215, 2.484027648895183, 3.0729824437283675,
#     3.6867651008068423, 4.249561942282852, 4.499886362039728, 4.648679453000878,
#     4.770534272304873, 4.844794139019431, 4.895192571939008, 4.903452318807486
# ]

# y_coords = [
#     0.0771409083181532, -0.03812136515210086, -0.20365307360840257, -0.31188336917609844,
#     -0.32441514900849056, -0.24365908050726467, -0.08159533572213637, 0.11020811938027594,
#     0.30929928255209327, 0.46050879627180497, 0.6157307261941061, 0.7930400439498658,
#     0.97357714286444, 1.1673830347471599, 1.3564903105985882, 1.5431376150004419,
#     1.7029885244787075, 1.7971200993308858, 1.8644922625372802, 1.9009758180380043,
#     1.9105316020730096, 2.060773043453525, 2.1722535650690924, 2.2149728761385203,
#     2.232170350125502, 2.236426519061847, 2.2337024047410488, 2.226976833579646,
#     2.2181189566838984, 2.192200078312105, 2.1578900642907572, 2.123291181496401,
#     2.0852699061600384, 2.0493498402367925, 2.033180674407799, 2.023373767703281,
#     2.0152965884132645, 2.010255339719507, 2.00692601069129, 2.0065051392060544
# ]

# # Ponto de referência
# ref_points_x = [0, 0, 0, 1, 2, 3, 4, 5]
# ref_points_y = [0, 1, 2, 2, 2, 2, 2, 2]

# # Coordenadas dos quadrados
# square_coords_x = [1, 2, 3, 4, 5.2]
# square_coords_y = [0.5, 1, 1.5, 1.75, 2]

# # Criando o gráfico
# plt.figure(figsize=(10, 6))
# plt.title("Real track X Ideal Path", fontsize=18)
# plt.plot(x_coords, y_coords, label='Real Track', marker='o', markersize=8, linestyle='-')
# plt.plot(ref_points_x, ref_points_y, color='red', label='Refrence Track', linewidth=3)
# plt.scatter(square_coords_x, square_coords_y, color='green', label='Solar Panels', marker='s', s=200)  # Adiciona quadrados
# plt.xlabel('X', fontsize=18)
# plt.ylabel('Y', fontsize=18)
# plt.legend(fontsize=18)
# # Aumentando o tamanho das fontes dos dados no eixo X e Y
# plt.gca().tick_params(axis='x', labelsize=18)  # Eixo X
# plt.gca().tick_params(axis='y', labelsize=18)  # Eixo Y
# plt.grid()
# plt.show()
