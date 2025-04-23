import os
import cv2
import numpy as np
import shutil
import random

# ========== FUNÇÕES UTILITÁRIAS ==========
enable_augmentations = True

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def read_image(path):
    image = cv2.imread(path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Erro ao ler: {path}")
        return None

def save_image(path, image, normalize=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if normalize:
        image = (image * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)

# ========== AUGMENTATIONS ==========

def random_scale(image, mask, scale_range=(0.9, 1.1)):
    scale = random.uniform(*scale_range)
    h, w = image.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    image_scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    return center_crop(image_scaled, mask_scaled, (h, w))

def random_translate(image, mask, max_shift=20):
    h, w = image.shape[:2]
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_translated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask_translated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return img_translated, mask_translated

def random_rotate(image, mask, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
    mask_rotated = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return img_rotated, mask_rotated

def horizontal_flip(image, mask):
    return cv2.flip(image, 1), cv2.flip(mask, 1)

def center_crop(image, mask, size=(480, 480)):
    h, w = image.shape[:2]
    ch, cw = size
    start_x = max(0, w // 2 - cw // 2)
    start_y = max(0, h // 2 - ch // 2)
    return image[start_y:start_y + ch, start_x:start_x + cw], mask[start_y:start_y + ch, start_x:start_x + cw]

def random_crop(img, mask, crop_size=(480, 480)):
    h, w = img.shape[:2]
    ch, cw = crop_size

    if h < ch or w < cw:
        # Se imagem for menor que o crop, redimensiona
        img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
        return img, mask

    start_x = random.randint(0, w - cw)
    start_y = random.randint(0, h - ch)

    img_crop = img[start_y:start_y + ch, start_x:start_x + cw]
    mask_crop = mask[start_y:start_y + ch, start_x:start_x + cw]
    return img_crop, mask_crop

def adjust_brightness(image, factor_range=(0.8, 1.2)):
    factor = random.uniform(*factor_range)
    image = np.clip(image * factor, 0, 255)
    return image.astype(np.uint8)

def adjust_contrast(image, factor_range=(0.8, 1.2)):
    factor = random.uniform(*factor_range)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def change_color_temperature(image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(image)
    shift = random.randint(-20, 20)
    r = np.clip(r + shift, 0, 255).astype(np.uint8)
    b = np.clip(b - shift, 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))

def apply_all_augmentations(img, mask):
    img, mask = random_scale(img, mask)
    img, mask = random_translate(img, mask)
    img, mask = random_rotate(img, mask)
    img, mask = horizontal_flip(img, mask)
    img, mask = random_crop(img, mask)

    if enable_augmentations:
        img = adjust_brightness(img)
        img = adjust_contrast(img)
        img = add_gaussian_noise(img)
        img = change_color_temperature(img)

    return img, mask

# ========== PROCESSAMENTOS ==========

def resize_images(img_dir, mask_dir, output_dir, num_images):
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
    total = min(len(img_files), len(mask_files), num_images)
    print(f"[RESIZE] {img_dir} → {output_dir} ({total} imagens)")

    for i in range(total):
        img = read_image(os.path.join(img_dir, img_files[i]))
        mask = read_image(os.path.join(mask_dir, mask_files[i]))
        if img is None or mask is None:
            continue

        img_resized = cv2.resize(img, (480, 480), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (480, 480), interpolation=cv2.INTER_NEAREST)

        save_image(os.path.join(output_dir, 'img', f"{i:04d}.png"), img_resized)
        save_image(os.path.join(output_dir, 'mask', f"{i:04d}.png"), mask_resized)
    print("✔ Redimensionamento concluído.")

def normalize_images(input_dir, output_dir, num_images):
    img_dir = os.path.join(input_dir, 'img')
    mask_dir = os.path.join(input_dir, 'mask')
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    total = min(len(img_files), len(mask_files), num_images)
    print(f"[NORMALIZE] {input_dir} → {output_dir} ({total} imagens)")

    for i in range(total):
        img = read_image(os.path.join(img_dir, img_files[i]))
        mask = read_image(os.path.join(mask_dir, mask_files[i]))
        if img is None or mask is None:
            continue

        img_norm = normalize_image(img)
        mask_norm = normalize_image(mask)

        save_image(os.path.join(output_dir, 'img', f"{i:04d}.png"), img_norm, normalize=True)
        save_image(os.path.join(output_dir, 'mask', f"{i:04d}.png"), mask_norm, normalize=True)
    print("✔ Normalização concluída.")

def affine_transform_images(input_dir, output_dir, num_images, num_augmented_versions=9):
    img_dir = os.path.join(input_dir, 'img')
    mask_dir = os.path.join(input_dir, 'mask')
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    total = min(len(img_files), len(mask_files), num_images)
    print(f"[AUGMENTATION] {input_dir} → {output_dir} ({total} imagens * {num_augmented_versions} versões)")

    for i in range(total):
        img = read_image(os.path.join(img_dir, img_files[i]))
        mask = read_image(os.path.join(mask_dir, mask_files[i]))
        if img is None or mask is None:
            continue

        for j in range(num_augmented_versions):
            img_aug, mask_aug = apply_all_augmentations(img, mask)
            idx = i * num_augmented_versions + j
            save_image(os.path.join(output_dir, 'img', f"{idx:05d}.png"), img_aug)
            save_image(os.path.join(output_dir, 'mask', f"{idx:05d}.png"), mask_aug)

    print("✔ Aumentações concluídas.")

def copy_images(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, start_index):
    img_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(src_mask_dir) if f.endswith('.png')])
    total = min(len(img_files), len(mask_files))

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    index = start_index
    for i in range(total):
        shutil.copy(os.path.join(src_img_dir, img_files[i]), os.path.join(dst_img_dir, f"{index:05d}.png"))
        shutil.copy(os.path.join(src_mask_dir, mask_files[i]), os.path.join(dst_mask_dir, f"{index:05d}.png"))
        index += 1

    return total, index

def unify_sets(splits):
    for split in splits:
        print(f"\n→ Unificando conjunto: {split.upper()}")

        dst_img_dir = f'dataset4/{split}/img'
        dst_mask_dir = f'dataset4/{split}/mask'
        start_index = 0
        total_geral = 0

        for src_type in ['resized', 'normalized', 'affined']:
            src_img_dir = f'dataset4/{src_type}/{split}/img'
            src_mask_dir = f'dataset4/{src_type}/{split}/mask'

            total, start_index = copy_images(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, start_index)
            total_geral += total
            print(f"   ✔ {src_type.capitalize():<10}: {total} imagens")

        print(f"✅ Total final em {split}: {total_geral} imagens")

    # Limpar pastas temporárias
    for folder in ['resized', 'normalized', 'affined']:
        shutil.rmtree(f'dataset4/{folder}', ignore_errors=True)
        print(f"🧹 Pasta removida: dataset4/{folder}")

# ========== EXECUÇÃO ==========

train_img_path = 'dataset4/artifitial'
train_mask_path = 'dataset4/artifitial'
val_img_path = 'dataset4/artifitial'
val_mask_path = 'dataset4/artifitial'

augmentation_number = 1.0
num_train = 9000
num_val = 1000

resize_images(train_img_path, train_mask_path, 'dataset4/artifitial/resized/train', num_train)
normalize_images('dataset4/artifitial/resized/train', 'dataset4/artifitial/normalized/train', num_train)
affine_transform_images('dataset4/artifitial/normalized/train', 'dataset4/artifitial/affined/train', num_train,num_augmented_versions=9)

resize_images(val_img_path, val_mask_path, 'dataset4/artifitial/resized/val', num_val)
normalize_images('dataset4/artifitial/resized/val', 'dataset4/artifitial/normalized/val', num_val)
affine_transform_images('dataset4/artifitial/normalized/val', 'dataset4/artifitial/affined/val', num_val, num_augmented_versions=9)

# unify_sets(['train', 'val'])
