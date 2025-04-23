import os

def renomear_imagens(pasta, prefixo="artifitial"):
    # Verifica se o caminho da pasta existe
    if not os.path.exists(pasta):
        print(f"Erro: O caminho '{pasta}' não existe!")
        return

    # Lista todos os arquivos na pasta
    arquivos = sorted([f for f in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, f))])
    
    # Verifica se há arquivos na pasta
    if not arquivos:
        print(f"Erro: Não há arquivos na pasta '{pasta}'")
        return

    # Exibe os arquivos encontrados para depuração
    print(f"Arquivos encontrados: {arquivos}")

    for i, nome_antigo in enumerate(arquivos):
        # Gera o novo nome com o prefixo 'artifitial' e numeração sequencial
        nome_novo = f"{prefixo}_{i:05d}{os.path.splitext(nome_antigo)[1]}"  # Mantém a extensão do arquivo
        # Caminhos completos para o arquivo antigo e novo
        caminho_antigo = os.path.join(pasta, nome_antigo)
        caminho_novo = os.path.join(pasta, nome_novo)
        
        # Renomeia o arquivo
        os.rename(caminho_antigo, caminho_novo)
        print(f"Renomeado: {nome_antigo} → {nome_novo}")
    
    print(f"✔ Renomeados {len(arquivos)} arquivos com o prefixo '{prefixo}'.")

# === Exemplo de uso ===
pasta_imagens = "dataset3/artifitial/val/img"
renomear_imagens(pasta_imagens)
