# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('train_data'):
    for filename in filenames:
        (os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from PIL import Image, ImageDraw
from IPython.display import display
import cv2
import os
import matplotlib.pyplot as plt

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Image processing
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects_info = []

    contour_image = np.zeros_like(mask)
    bounding_rect_image = mask.copy()
    yolo_bbox_image = mask.copy()
    
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        class_label = 0 
        x_center, y_center, normalized_width, normalized_height = convert_coordinates_to_yolo(mask.shape[1], mask.shape[0], x, y, width, height)
        objects_info.append((class_label, x_center, y_center, normalized_width, normalized_height))
        
        cv2.rectangle(bounding_rect_image, (x, y), (x + width, y + height), 255, thickness=2)

        cv2.drawContours(contour_image, [contour], 0, 255, thickness=2)

    display(Image.fromarray(mask)) #display mask
    display(Image.fromarray(contour_image))# display contours
    display(Image.fromarray(bounding_rect_image)) #display rectangle

    return objects_info

def convert_coordinates_to_yolo(image_width, image_height, x, y, width, height):
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    return x_center, y_center, normalized_width, normalized_height

def write_yolo_annotations(output_path, image_name, objects_info):
    annotation_file_path = os.path.join(output_path, image_name)

    with open(annotation_file_path, "w") as file:
        for obj_info in objects_info:
            line = f"{obj_info[0]} {obj_info[1]} {obj_info[2]} {obj_info[3]} {obj_info[4]}\n"
            file.write(line)


def read_images(image_path=None, image_file = None):
    image_path = os.path.join(image_path, image_file)
    image = cv2.imread(image_path)
    if image is not None:
        return image
    else:
        print(f"Erro ao ler imagem: {image_path}")



def organize_train_files(selected_imgs, img, img_path):
    #faz um laço for para a leitura das imagens de treino
    for image_file in selected_imgs:
        img.append(read_images(img_path, image_file))  


input_path = "dataset3/train/Masks"
output_path = "dataset3/train/labels"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)


# Process each mask in the kaggle input directory
for mask_name in os.listdir(input_path):
    mask_path = os.path.join(input_path, mask_name)
    image_name = os.path.basename(mask_path).replace(".png", ".txt") #change the file type to txt
    objects_info = process_mask(mask_path)
    write_yolo_annotations(output_path, image_name, objects_info)


print("Indo para validação\n\n")

input_path = "dataset3/val/Masks"
output_path = "dataset3/val/labels"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Process each mask in the kaggle input directory
for mask_name in os.listdir(input_path):
    mask_path = os.path.join(input_path, mask_name)
    image_name = os.path.basename(mask_path).replace(".png", ".txt") #change the file type to txt
    objects_info = process_mask(mask_path)
    write_yolo_annotations(output_path, image_name, objects_info)