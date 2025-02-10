import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image, ImageDraw
from IPython.display import display

def read_images(image_path=None, image_file = None):
    image_path = os.path.join(image_path, image_file)
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    if image is not None:
        return image
    else:
        print(f"Erro ao ler imagem: {image_path}")

def organize_train_files(selected_imgs, selected_masks, img, mask, img_path, mask_path, num):
    #faz um laço for para a leitura das imagens de treino
    for image_file in selected_imgs[:num]:
        img.append(read_images(img_path, image_file))  
    
    for image_file in selected_masks[:num]:
        mask.append(read_images(mask_path, image_file))  

def organize_test_files(selected_imgs, selected_masks, img, mask, img_path, mask_path, num):
    #faz um laço for para a leitura das imagens de treino
    for image_file in selected_imgs[:num]:
        img.append(read_images(img_path, image_file))  
    
    for image_file in selected_masks[:num]:
        mask.append(read_images(mask_path, image_file))  

def normalize_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    min_val = np.min(gray_img)
    max_val = np.max(gray_img)
    normalized_image = (gray_img - min_val) / (max_val - min_val)
    normalized_image_255 = (normalized_image * 255).astype(np.uint8)
    return normalized_image_255

def flip_image(img_2_flip, img, direction):
    img_2_flip.append(cv2.flip(img, direction))

def affine_image(img_2_affine, img):
    # Dimensões da imagem
    rows, cols, ch= img.shape

    # Definir pontos para a transformação afim
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    # Calcular a matriz de transformação afim
    M = cv2.getAffineTransform(pts1, pts2)

    # Aplicar a transformação afim
    affine_transformed_image = cv2.warpAffine(img, M, (cols, rows))

    # Converter a imagem transformada de BGR para RGB para exibição
    affine_transformed_image_rgb = cv2.cvtColor(affine_transformed_image, cv2.COLOR_BGR2RGB)

    img_2_affine.append(affine_transformed_image_rgb)

#defino o caminho das imagens de treinamento e de teste/vallidacao
train_imgs_directory_path = 'imagens_gazebo'
train_masks_directory_path = 'imagens_gazebo'
val_imgs_directory_path = 'imagens_gazebo'
val_mask_directory_path = 'imagens_gazebo'

#defino a quantidade de imagens que serao processadas
number = 20
num_images_train_2_read = 20
num_images_val_2_read = 0

#lista todas as imagens
all_train_imgs_files = os.listdir(train_imgs_directory_path)
all_train_masks_files = os.listdir(train_masks_directory_path)
all_val_imgs_files = os.listdir(val_imgs_directory_path)
all_val_masks_files = os.listdir(val_mask_directory_path)

print("List completo")

#armazeno numa lista todos os arquivos de imagem
train_imgs_files = [f for f in all_train_imgs_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
train_masks_files = [f for f in all_train_masks_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
val_imgs_files = [f for f in all_val_imgs_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
val_masks_files = [f for f in all_val_masks_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

print("Armazenamento completo")

#ordenar a lista de arquivos de imagem
train_imgs_files.sort()
train_masks_files.sort()
val_imgs_files.sort()
val_masks_files.sort()

print("Sort completo")


#seleciona numa lista ateh 200 imagens
selected_train_imgs = train_imgs_files
selected_train_masks = train_masks_files
selected_val_imgs = val_imgs_files
selected_val_masks = val_masks_files

#leitura das imagens usando OpenCV

train_images = []
train_masks = []
val_images = []
val_masks = []
val_images_normalized = [None] * num_images_train_2_read
val_masks_mormalized = [None] * num_images_train_2_read
val_images_affined = []
val_masks_affined = []
val_images_flipped = []
val_masks_flipped = []
train_images_normalized = [None] * num_images_train_2_read
train_masks_mormalized = [None] * num_images_train_2_read
train_images_affined = []
train_masks_affined = []
train_images_flipped = []
train_masks_flipped = []

organize_train_files(selected_train_imgs, selected_train_masks, train_images, train_masks, train_imgs_directory_path, train_masks_directory_path, num_images_train_2_read)

print(f"{len(train_images)} imagens de treino lidas com sucesso")
# print(f"{len(train_masks)} masks de treino lidas com sucesso")

organize_test_files(selected_val_imgs, selected_val_masks, val_images, val_masks, val_imgs_directory_path, val_mask_directory_path, num_images_val_2_read)

print(f"{len(val_images)} imagens de teste lidas com sucesso")
# print(f"{len(val_masks)} imagens de mask lidas com sucesso")

#redimensionando todas as imagens
# for images in range(len(train_images)):
for images in range(num_images_train_2_read):
    train_images[images] = cv2.resize(train_images[images], (480, 480),interpolation=cv2.INTER_LINEAR)
    # train_masks[images] = cv2.resize(train_masks[images], (480, 480),interpolation=cv2.INTER_LINEAR)
    train_images_normalized[images] = normalize_image(train_images[images])
    # train_masks_mormalized[images] = normalize_image(train_masks[images])
    
    affine_image(train_images_affined, train_images[images])
    # affine_image(train_masks_affined, train_masks[images])

    flip_image(train_images_flipped, train_images[images], 0)
    # flip_image(train_masks_flipped, train_masks[images], 0)


print("Images normalizadas")


# for images in range(len(val_images)):
for images in range(num_images_val_2_read):
    val_images[images] = cv2.resize(val_images[images], (480, 480),interpolation=cv2.INTER_LINEAR)
    # val_masks[images] = cv2.resize(val_masks[images], (480, 480),interpolation=cv2.INTER_LINEAR)
    val_images_normalized[images] = normalize_image(val_images[images])
    # val_masks_mormalized[images] = normalize_image(val_masks[images])

    affine_image(val_images_affined, val_images[images])
    # affine_image(val_masks_affined, val_masks[images])
    flip_image(val_images_flipped, val_images[images], 0)
    # flip_image(val_masks_flipped, val_masks[images], 0)


print("Masks normalizadas")

min_len = min(len(train_imgs_files), len(train_images), len(train_images_affined), len(train_images_flipped), len(train_images_normalized))
print(min_len)

# Save images with the same original name
for i in range(min_len):
    image_name = selected_train_imgs[i]
    image_affined_name = f"affined_{i}.png"
    image_flipped_name = f"flipped_{i}.png"
    image_normalized_name = f"normalized_{i}.png"
    # mask_name = selected_train_masks[i]
    image = train_images[i]
    image_affined = train_images_affined[i]
    image_flipped = train_images_flipped[i]
    image_normalized = train_images_normalized[i]
    # masks = train_masks[i]

    cv2.imwrite(os.path.join(train_imgs_directory_path, image_name), (image * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(train_imgs_directory_path, image_affined_name), (image_affined * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(train_imgs_directory_path, image_flipped_name), (image_flipped * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(train_imgs_directory_path, image_normalized_name), (image_normalized * 255).astype(np.uint8))
    # cv2.imwrite(os.path.join(train_masks_directory_path, mask_name), (masks * 255).astype(np.uint8))
    print(f"Imagem reescrita em: {os.path.join(train_imgs_directory_path, image_name)}")
    # print(f"Imagem reescrita em: {os.path.join(train_masks_directory_path, mask_name)}")

# min_len = min(len(val_imgs_files), len(val_images), len(val_images_affined), len(val_images_flipped), len(val_images_normalized))

# print(min_len)

# for i in range(min_len):
#     image_val_name = selected_val_imgs[i]
#     image_affined_val_name = f"affined_val_{i}.png"
#     image_flipped_val_name = f"flipped_val_{i}.png"
#     image_normalized_val_name = f"normalized_val_{i}.png"
#     # mask_val_name = selected_val_masks[i]
#     image_val = val_images[i]
#     image_val_affined = val_images_affined[i]
#     image_val_flipped = val_images_flipped[i]
#     image_val_normalized = val_images_normalized[i]
#     # masks_val = val_masks[i]
#     cv2.imwrite(os.path.join(val_imgs_directory_path, image_val_name), (image_val * 255).astype(np.uint8))
#     cv2.imwrite(os.path.join(val_imgs_directory_path, image_affined_val_name), (image_val_affined * 255).astype(np.uint8))
#     cv2.imwrite(os.path.join(val_imgs_directory_path, image_flipped_val_name), (image_val_flipped * 255).astype(np.uint8))
#     cv2.imwrite(os.path.join(val_imgs_directory_path, image_normalized_val_name), (image_val_normalized * 255).astype(np.uint8))
#     # cv2.imwrite(os.path.join(val_mask_directory_path, mask_val_name), (masks_val * 255).astype(np.uint8))
#     print(f"Imagem reescrita em: {os.path.join(val_imgs_directory_path, image_val_name)}")
#     # print(f"Imagem reescrita em: {os.path.join(val_mask_directory_path, mask_val_name)}")


#faz o plot de todas as imagens
# for i in range(num_images_train_2_read):
#     plt.figure()
#     plt.imshow(train_images[i])
#     plt.title(f"Imagem de treino {i+1}")
#     plt.axis('off')

#     # cv2.imshow("Image", train_images[i])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


#     # plt.figure()
#     # plt.imshow(train_masks[i])
#     # plt.title(f"Mascara de treino {i+1}")
#     # plt.axis('off')

#     plt.figure()
#     plt.imshow(train_images_affined[i])
#     plt.title(f"Imagem de treino affined {i+1}")
#     plt.axis('off')

#     # plt.figure()
#     # plt.imshow(train_masks_affined[i])
#     # plt.title(f"Mascara de treino affined {i+1}")
#     # plt.axis('off')
    
#     plt.figure()
#     plt.imshow(train_images_flipped[i])
#     plt.title(f"Imagem de treino flipped {i+1}")
#     plt.axis('off')

#     # plt.figure()
#     # plt.imshow(train_masks_flipped[i])
#     # plt.title(f"Mascara de treino flipped {i+1}")
#     # plt.axis('off')
  
#     plt.figure()
#     plt.imshow(train_images_normalized[i])
#     plt.title(f"Imagem de treino normalizada {i+1}")
#     plt.axis('off')
    
#     # plt.figure()
#     # plt.imshow(train_masks_mormalized[i])
#     # plt.title(f"Mascara de treino normalizada {i+1}")
#     # plt.axis('off')

    
#     if i >  num_images_val_2_read:
#         plt.show()
#         continue
    
#     else:
#         plt.figure()
#         plt.imshow(val_images[i])
#         plt.title(f"Imagem de teste {i+1}")
#         plt.axis('off')

#         # plt.figure()
#         # plt.imshow(val_masks[i])
#         # plt.title(f"Mascara de teste {i+1}")
#         # plt.axis('off')
#         # plt.axis('off')
        
#         plt.figure()
#         plt.imshow(val_images_affined[i])
#         plt.title(f"Imagem de teste affined {i+1}")
#         plt.axis('off')
#         plt.axis('off')

#         # plt.figure()
#         # plt.imshow(val_masks_affined[i])
#         # plt.title(f"Mascara de teste affined {i+1}")
#         # plt.axis('off')
#         # plt.axis('off')

#         plt.figure()
#         plt.imshow(val_images_flipped[i])
#         plt.title(f"Imagem de teste flipped {i+1}")
#         plt.axis('off')
#         plt.axis('off')

#         # plt.figure()
#         # plt.imshow(val_masks_flipped[i])
#         # plt.title(f"Mascara de teste flipped {i+1}")
#         # plt.axis('off')
#         # plt.axis('off')

#         plt.figure()
#         plt.imshow(val_images_normalized[i])
#         plt.title(f"Imagem de teste normalizada {i+1}")
#         plt.axis('off')
#         plt.axis('off')

#         # plt.figure()
#         # plt.imshow(val_masks_mormalized[i])
#         # plt.title(f"Mascara de teste normalizada{i+1}")
#         # plt.axis('off')
#         plt.show()
