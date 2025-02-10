#! /usr/bin/env python

import cv2
import os

# Caminho para a pasta e nome do arquivo da imagem
folder_path = '/home/johann/CNN_Model'
image_file = 'imagem3.png'
image_path = os.path.join(folder_path, image_file)

cv_image = cv2.imread(image_path)
image = cv2.imread(image_path)

while(True):

    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(1,1),0)

    # Realiza a detecção de bordas utilizando Canny
    edges = cv2.Canny(blur, 80, 120)

    # Encontra os contornos na imagem de bordas
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop sobre todos os contornos encontrados
    for contour in contours:
        # Calcula os momentos do contorno
        M = cv2.moments(contour)

        # Verifica se o contorno é válido para evitar divisão por zero
        if M["m00"] != 0:
            # Calcula as coordenadas do centroide
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Desenha o centroide na imagem colorida
            cv2.circle(cv_image, (cX, cY), 5, (0, 255, 0), -1)

    edges_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    cv2.imshow("Edge window", edges_image)
    cv2.waitKey(3)

    # cv2.imshow("Gaussian window", blur)
    # cv2.waitKey(3)
