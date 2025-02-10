import os
import subprocess
import shutil

# Caminhos
python_exe = r"c:\Users\johan\AppData\Local\Programs\Python\Python311\python.exe"
yolo_script = r"yolov5/detect.py"
weights = r"yolov5/runs/train/yolov5m_dataset3_30_epochs/weights/best.pt"
device = "0"
source_dir = r".\Imagens_gazebo\."
output_dir = r".\Detect.\yolov5m_dataset3_30epochs.\processed_images"  # Diretório para salvar as imagens processadas

# Extensões de imagens que você deseja processar
valid_extensions = [".png", ".jpg", ".jpeg"]

# Número máximo de imagens a serem processadas
max_images = 20  # Substitua com o número desejado

# Função para processar cada imagem no diretório
def process_images_in_directory(directory, max_images, output_directory):
    # Cria o diretório de destino, se não existir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    processed_count = 0
    for filename in os.listdir(directory):
        if processed_count >= max_images:
            break
        # Verifica se o arquivo tem uma extensão válida
        if any(filename.endswith(ext) for ext in valid_extensions):
            file_path = os.path.join(directory, filename)
            command = [
                python_exe,
                yolo_script,
                "--weights", weights,
                "--device", device,
                "--source", file_path,
                "--save-txt",  # Caso você queira salvar os resultados em um arquivo de texto
                "--save-conf",  # Caso você queira salvar as confiabilidades das detecções
                "--project", output_directory,  # Define o diretório onde o YOLO salva os resultados
                "--name", "resultados",  # Nome da subpasta onde serão salvos os resultados
                "--exist-ok"  # Sobrescreve a pasta se ela já existir
            ]
            
            # Executa o comando
            print(f"Processing {file_path}...")
            subprocess.run(command)
            print(f"Finished processing {file_path}")
            
            processed_count += 1

# Executa a função
process_images_in_directory(source_dir, max_images, output_dir)
