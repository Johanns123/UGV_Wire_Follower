import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    x_values = []
    y_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if 'x=' in line and 'y=' in line:
                # Extraindo os valores de x e y da linha
                parts = line.strip().split(', ')
                x = float(parts[0].split('=')[1])
                y = float(parts[1].split('=')[1])
                x_values.append(x)
                y_values.append(y)
    
    return x_values, y_values

def plot_data(x_values, y_values):
    plt.figure(figsize=(10, 6))
    plt.title("Real track X Wire Position", fontsize=18)
    plt.xlabel('X[m]', fontsize=18)
    plt.ylabel('Y[m]', fontsize=18)
    # Aumentando o tamanho das fontes dos dados no eixo X e Y
    plt.gca().tick_params(axis='x', labelsize=18)  # Eixo X
    plt.gca().tick_params(axis='y', labelsize=18)  # Eixo Y
    plt.grid()
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label = 'Robot')
     # Adicionando o percurso quadrado
    square_x = [4, 4, 12, 12, 4]  # Coordenadas x dos vértices do quadrado
    square_y = [12, 40, 40, 12, 12]  # Coordenadas y dos vértices do quadrado
    plt.plot(square_x, square_y, marker='s', linestyle='-', color='r', label='Cable')
    plt.legend(fontsize=18, loc='upper right')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Substitua 'data.txt' pelo caminho do seu arquivo
    file_path = 'wire_tracking.txt'
    

    try:
        x_values, y_values = read_data_from_file(file_path)
        if x_values and y_values:
            plot_data(x_values, y_values)
        else:
            print("Nenhum dado válido encontrado no arquivo.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")
