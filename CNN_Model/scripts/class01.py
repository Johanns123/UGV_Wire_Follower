#%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.pyplot as mping
import cv2
import numpy as np


img = cv2.imread("passaros.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

R, G, B = cv2.split(img_rgb)

#Criar imagens vazias para cada canals
zeros = np.zeros_like(R)

# Combinar canais para exibir apenas uma cor por vez
only_red = cv2.merge([R, zeros, zeros])
only_green = cv2.merge([zeros, G, zeros])
only_blue = cv2.merge([zeros, zeros, B])

# Exibir as imagens
plt.figure(figsize=(10, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(only_red)
plt.title('Apenas Vermelho')

plt.subplot(1, 4, 3)
plt.imshow(only_green)
plt.title('Apenas Verde')

plt.subplot(1, 4, 4)
plt.imshow(only_blue)
plt.title('Apenas Azul')

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_hsv, cmap='hsv')
plt.title('Imagem em HSV')

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap='gray',)
plt.title('Imagem em escala de cinza')

plt.show()

from PIL import Image
import urllib.request
import cv2

# URL da imagem
url = 'https://matplotlib.org/3.3.3/_images/stinkbug.png'

# Abrir a URL e ler a imagem
with urllib.request.urlopen(url) as url_response:
    image = np.array(Image.open(url_response))

# Converter a imagem de RGB (Pillow) para BGR (OpenCV)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Converter a imagem de BGR para HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Mapear e exibir os canais HSV separadamente
hue, saturation, value = cv2.split(image_hsv)

# Exibir o canal Hue
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Hue Channel')
plt.imshow(hue, cmap='hsv')
plt.axis('off')

# Exibir o canal Saturation
plt.subplot(1, 3, 2)
plt.title('Saturation Channel')
plt.imshow(saturation, cmap='gray')
plt.axis('off')

# Exibir o canal Value
plt.subplot(1, 3, 3)
plt.title('Value Channel')
plt.imshow(value, cmap='gray')
plt.axis('off')

plt.show()

plt.imshow(image_hsv)
plt.axis('off')  # Ocultar os eixos
plt.show()

# Converter a imagem de BGR para Grayscale
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Exibir a imagem em Grayscale usando matplotlib
plt.imshow(image_gray, cmap='gray')
plt.axis('off')  # Ocultar os eixos
plt.title('Grayscale Image')
plt.show()


# Definir novas dimensões (por exemplo, 200x200 pixels)
new_width = 200
new_height = 200

# Redimensionar a imagem
resized_image = cv2.resize(image_bgr, (new_width, new_height))

# Converter a imagem redimensionada de BGR para RGB para exibição
resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# Exibir a imagem redimensionada
plt.imshow(resized_image_rgb)
plt.axis('off')  # Ocultar os eixos
plt.title('Resized Image')
plt.show()


# Dimensões da imagem
rows, cols, ch = image_bgr.shape

# Definir pontos para a transformação afim
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# Calcular a matriz de transformação afim
M = cv2.getAffineTransform(pts1, pts2)

# Aplicar a transformação afim
affine_transformed_image = cv2.warpAffine(image_bgr, M, (cols, rows))

# Converter a imagem transformada de BGR para RGB para exibição
affine_transformed_image_rgb = cv2.cvtColor(affine_transformed_image, cv2.COLOR_BGR2RGB)

# Exibir a imagem original e a transformada lado a lado
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Affine Transformed Image')
plt.imshow(affine_transformed_image_rgb)
plt.axis('off')

plt.show()

# Aplicar o detector de bordas de Canny
edges = cv2.Canny(image_gray, threshold1=100, threshold2=200)

# Exibir a imagem original e a imagem com detecção de bordas lado a lado
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()

# Verificar se SIFT está disponível
if cv2.__version__.startswith('4'):
    sift = cv2.SIFT_create()
else:
    sift = cv2.xfeatures2d.SIFT_create()

# URL da imagem
url = 'https://matplotlib.org/3.3.3/_images/stinkbug.png'

# Abrir a URL e ler a imagem
with urllib.request.urlopen(url) as url_response:
    image = np.array(Image.open(url_response))

# Converter a imagem de RGB (Pillow) para BGR (OpenCV)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Converter a imagem para escala de cinza (necessário para o SIFT)
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Detectar keypoints e calcular descritores
keypoints, descriptors = sift.detectAndCompute(image_gray, None)

# Desenhar keypoints na imagem
image_with_keypoints = cv2.drawKeypoints(image_bgr, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Converter a imagem de BGR para RGB para exibição
image_with_keypoints_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

# Exibir a imagem original e a imagem com keypoints lado a lado
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image with SIFT Keypoints')
plt.imshow(image_with_keypoints_rgb)
plt.axis('off')

plt.show()