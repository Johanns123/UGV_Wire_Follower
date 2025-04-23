import os
import cv2
import numpy as np
import shutil
import random

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

# === Funções de Augmentation ===
def rotate(img, mask, angle=10):
    M = cv2.getRotationMatrix2D((240, 240), angle, 1.0)
    return cv2.warpAffine(img, M, (480, 480)), cv2.warpAffine(mask, M, (480, 480))

def translate(img, mask, tx=20, ty=20):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (480, 480)), cv2.warpAffine(mask, M, (480, 480))

def scale(img, mask, fx=1.2, fy=1.2):
    return cv2.resize(img, None, fx=fx, fy=fy), cv2.resize(mask, None, fx=fx, fy=fy)

def flip(img, mask):
    return cv2.flip(img, 1), cv2.flip(mask, 1)

def random_crop(img, mask, crop_size=(400, 400)):
    h, w, _ = img.shape
    ch, cw = crop_size
    if h < ch or w < cw:
        return img, mask  # Pula se for menor
    start_x = random.randint(0, w - cw)
    start_y = random.randint(0, h - ch)
    return img[start_y:start_y+ch, start_x:start_x+cw], mask[start_y:start_y+ch, start_x:start_x+cw]

def adjust_brightness(img, value=30):
    return cv2.convertScaleAbs(img, beta=value)

def adjust_contrast(img, alpha=1.5):
    return cv2.convertScaleAbs(img, alpha=alpha)

def add_gaussian_noise(img, mean=0, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def change_color_temperature(img, delta_b=30, delta_r=-30):
    b, g, r = cv2.split(img)
    b = cv2.add(b, delta_b)
    r = cv2.add(r, delta_r)
    return cv2.merge((b, g, r))

def shear(img, mask, shear_factor=0.2):
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0])), cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

def perspective_transform(img, mask):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    pts2 = np.float32([[10,10], [w-10,0], [0,h], [w-20,h-10]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w,h)), cv2.warpPerspective(mask, M, (w,h))

def elastic_deformation(img, mask, alpha=1000, sigma=40):
    random_state = np.random.RandomState(None)
    shape = img.shape[:2]

    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17,17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17,17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    img_deformed = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    mask_deformed = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST)
    return img_deformed, mask_deformed

def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8"))
    return cv2.LUT(img, table)

def hue_saturation_shift(img, hue_shift=10, sat_shift=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[..., 0] = np.clip(hsv[..., 0] + hue_shift, 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1] + sat_shift, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def salt_and_pepper_noise(img, amount=0.01):
    output = np.copy(img)
    num_salt = np.ceil(amount * img.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * img.size * 0.5).astype(int)

    coords = tuple([np.random.randint(0, i - 1, num_salt) for i in img.shape])
    output[coords] = 255

    coords = tuple([np.random.randint(0, i - 1, num_pepper) for i in img.shape])
    output[coords] = 0
    return output

def motion_blur(img, size=15):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)

def jpeg_artifacts(img, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def grid_distortion(img, mask, num_steps=5, distort_limit=0.3):
    h, w = img.shape[:2]
    step_x = w // num_steps
    step_y = h // num_steps

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            dx = random.uniform(-distort_limit, distort_limit) * step_x
            dy = random.uniform(-distort_limit, distort_limit) * step_y
            map_x[i, j] = j + dx
            map_y[i, j] = i + dy

    img_distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    mask_distorted = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST)
    return img_distorted, mask_distorted

def mixup(img1, img2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    return (lam * img1 + (1 - lam) * img2).astype(np.uint8)

def cutmix(img1, img2):
    h, w = img1.shape[:2]
    cx, cy = random.randint(0, w), random.randint(0, h)
    cut_w, cut_h = w // 2, h // 2
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    img1_copy = img1.copy()
    img1_copy[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    return img1_copy

def apply_all_augmentations(img, mask):
    return [
        ("rotate", *rotate(img, mask, angle=15)),
        ("translate", *translate(img, mask, tx=20, ty=20)),
        ("scale", *scale(img, mask, fx=1.2, fy=1.2)),
        ("flip", *flip(img, mask)),
        ("crop", *random_crop(img, mask)),
        ("brightness", adjust_brightness(img), mask),
        ("contrast", adjust_contrast(img), mask),
        ("noise", add_gaussian_noise(img), mask),
        ("color_temp", change_color_temperature(img), mask),
        ("shear", *shear(img, mask)),
        ("perspective", *perspective_transform(img, mask)),
        ("elastic", *elastic_deformation(img, mask)),
        ("gamma", gamma_correction(img), mask),
        ("hue_sat", hue_saturation_shift(img), mask),
        ("salt_pepper", salt_and_pepper_noise(img), mask),
        ("motion_blur", motion_blur(img), mask),
        ("jpeg_artifacts", jpeg_artifacts(img), mask),
        ("grid_distortion", *grid_distortion(img, mask)),
    ]


def affine_transform_images(input_dir, output_dir, num_images, only_affine=False):
    img_dir = os.path.join(input_dir, 'img')
    mask_dir = os.path.join(input_dir, 'mask')
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    total = min(len(img_files), len(mask_files), num_images)
    print(f"[AUGMENTATION] {input_dir} → {output_dir} ({total} imagens base)")

    idx = 0
    for i in range(total):
        img = read_image(os.path.join(img_dir, img_files[i]))
        mask = read_image(os.path.join(mask_dir, mask_files[i]))
        if img is None or mask is None:
            continue

        if only_affine:
            img_aug, mask_aug = rotate(img, mask, angle=10)
            save_image(os.path.join(output_dir, 'img', f"{idx:05d}.png"), img_aug)
            save_image(os.path.join(output_dir, 'mask', f"{idx:05d}.png"), mask_aug)
            idx += 1
        else:
            augmentations = apply_all_augmentations(img, mask)
            for aug_name, img_aug, mask_aug in augmentations:
                img_aug = cv2.resize(img_aug, (480, 480))
                mask_aug = cv2.resize(mask_aug, (480, 480))
                save_image(os.path.join(output_dir, 'img', f"{idx:05d}.png"), img_aug)
                save_image(os.path.join(output_dir, 'mask', f"{idx:05d}.png"), mask_aug)
                idx += 1
    print(f"✔ Augmentation concluída: {idx} imagens geradas.")

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

        dst_img_dir = f'dataset3/artifitial/{split}/img'
        dst_mask_dir = f'dataset3/artifitial/{split}/mask'
        start_index = 0
        total_geral = 0

        for src_type in ['resized', 'normalized', 'affined']:
            src_img_dir = f'dataset3/{src_type}/{split}/img'
            src_mask_dir = f'dataset3/{src_type}/{split}/mask'
            if not os.path.exists(src_img_dir):
                continue
            total, start_index = copy_images(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, start_index)
            total_geral += total
            print(f"   ✔ {src_type.capitalize():<10}: {total} imagens")

        print(f"✅ Total final em {split}: {total_geral} imagens")

    # Limpar pastas intermediárias
    for folder in ['resized', 'normalized', 'affined']:
        shutil.rmtree(f'dataset3/{folder}', ignore_errors=True)
        print(f"🗑️  Pasta {folder} apagada.")

# === Configurações ===
only_affine = False  # → Altere para True se quiser só transformação afim

train_img_path = '../Imagens_gazebo/train'
train_mask_path = '../Imagens_gazebo/train'
val_img_path = '../Imagens_gazebo/val'
val_mask_path = '../Imagens_gazebo/val'

num_train = 391
num_val = 289

# === Execução ===
resize_images(train_img_path, train_mask_path, 'dataset3/resized/train', num_train)
normalize_images('dataset3/resized/train', 'dataset3/normalized/train', num_train)
affine_transform_images('dataset3/normalized/train', 'dataset3/affined/train', num_train, only_affine)

resize_images(val_img_path, val_mask_path, 'dataset3/resized/val', num_val)
normalize_images('dataset3/resized/val', 'dataset3/normalized/val', num_val)
affine_transform_images('dataset3/normalized/val', 'dataset3/affined/val', num_val, only_affine)

unify_sets(['train', 'val'])
