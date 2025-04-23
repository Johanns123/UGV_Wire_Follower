import os

# Caminho da pasta com as imagens
pasta = '../dataset4/val/images'

# Lista e ordena os arquivos .png
arquivos = sorted([f for f in os.listdir(pasta) if f.endswith('.png')])

# Renomeia para formato com 5 dígitos: 00000.png, 00001.png, ...
for idx, nome_original in enumerate(arquivos):
    novo_nome = f"{idx:05}.png"
    caminho_antigo = os.path.join(pasta, nome_original)
    caminho_novo = os.path.join(pasta, novo_nome)
    os.rename(caminho_antigo, caminho_novo)

print(f"Renomeados {len(arquivos)} arquivos com sucesso!")
