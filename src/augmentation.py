#!/usr/bin/env python3
"""
Augmentation script for Assignment 2.

Applies 5 different distortions to each original PNG image to simulate
things like scanner noise, compression, low-res scanning, etc.

Each image gets 5 augmented copies, so the total dataset becomes 6x
the original size.
"""

import os
import random
import numpy as np
import cv2
from PIL import Image

SUBDIRS = ['word_pdfs_png', 'google_docs_pdfs_png', 'python_pdfs_png']


# cv2.imread doesn't work with special characters in filenames on Windows
# (eg Galápagos_Islands.png). Found this workaround online - read the file
# as raw bytes first then decode it.
def cv2_read(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

def cv2_save_png(path, img):
    ok, buf = cv2.imencode('.png', img)
    if ok:
        buf.tofile(path)

def cv2_save_jpg(path, img, quality):
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        buf.tofile(path)


# --- Augmentation functions ---

def add_gaussian_noise(image_path, output_path):
    """Add random gaussian noise (sigma between 5 and 20)"""
    img = cv2_read(image_path)
    if img is None:
        return
    sigma = random.uniform(5, 20)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = cv2.add(img.astype(np.float32), noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    cv2_save_png(output_path, noisy)


def apply_jpeg_compression(image_path, output_path):
    """Compress with random JPEG quality (20-80)"""
    img = cv2_read(image_path)
    if img is None:
        return
    quality = random.randint(20, 80)
    cv2_save_jpg(output_path, img, quality)


def apply_dpi_downsampling(image_path, output_path):
    """Simulate going from 300dpi to 150 or 72 dpi"""
    img = Image.open(image_path).convert('L')
    scale = random.choice([0.5, 0.24])  # 150dpi or 72dpi
    small = img.resize(
        (int(img.width * scale), int(img.height * scale)), Image.BILINEAR
    )
    # scale back up so the dimensions stay consistent for classification
    restored = small.resize((img.width, img.height), Image.NEAREST)
    restored.save(output_path, 'PNG')


def apply_random_crop(image_path, output_path):
    """Crop 1-3% off each border then resize back"""
    img = Image.open(image_path).convert('L')
    w, h = img.size
    pct = random.uniform(0.01, 0.03)
    left = int(w * pct)
    top = int(h * pct)
    right = w - int(w * pct)
    bottom = h - int(h * pct)
    cropped = img.crop((left, top, right, bottom))
    # resize back to keep same dimensions
    restored = cropped.resize((w, h), Image.BILINEAR)
    restored.save(output_path, 'PNG')


def apply_bit_depth_reduction(image_path, output_path):
    """Drop from 8-bit (256 levels) to 4-bit (16 levels)"""
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    arr = (arr // 16) * 16  # quantize
    Image.fromarray(arr.astype(np.uint8), mode='L').save(output_path, 'PNG')


def process_folder(input_dir, output_dir, folder_name):
    """Run all 5 augmentations on every PNG in one folder."""
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    print(f"\n[{folder_name}] Found {len(images)} original images, generating {len(images) * 5} augmented...")

    for i, img_file in enumerate(images, 1):
        in_path = os.path.join(input_dir, img_file)
        base = img_file.replace('.png', '')

        try:
            # gaussian noise
            add_gaussian_noise(in_path,
                os.path.join(output_dir, f"{base}_aug_gaussian.png"))

            # jpeg compression - need to save as jpg first then convert to png
            jpg_tmp = os.path.join(output_dir, f"{base}_aug_jpeg.jpg")
            apply_jpeg_compression(in_path, jpg_tmp)
            Image.open(jpg_tmp).save(
                os.path.join(output_dir, f"{base}_aug_jpeg.png"))
            os.remove(jpg_tmp)

            # dpi downsampling
            apply_dpi_downsampling(in_path,
                os.path.join(output_dir, f"{base}_aug_dpi.png"))

            # random crop
            apply_random_crop(in_path,
                os.path.join(output_dir, f"{base}_aug_crop.png"))

            # bit depth reduction
            apply_bit_depth_reduction(in_path,
                os.path.join(output_dir, f"{base}_aug_bitdepth.png"))

            if i % 50 == 0:
                print(f"  Progress: {i}/{len(images)} images processed...")

        except Exception as e:
            print(f"  Error augmenting {img_file}: {e}")

    # quick sanity check
    aug_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    expected = len(images) * 5
    status = "OK" if aug_count == expected else f"WARNING expected {expected}"
    print(f"  [{folder_name}] Done - {aug_count} augmented images [{status}]")
    return len(images), aug_count


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_base = os.path.join(base_dir, 'data', 'original_pdfs')
    output_base = os.path.join(base_dir, 'data', 'augmented_images')

    print("=" * 60)
    print("Dataset Augmentation")
    print(f"Input:  {input_base}")
    print(f"Output: {output_base}")
    print("=" * 60)

    total_orig = 0
    total_aug = 0

    for subdir in SUBDIRS:
        in_dir = os.path.join(input_base, subdir)
        out_dir = os.path.join(output_base, subdir)

        if not os.path.exists(in_dir):
            print(f"\nWARNING: {in_dir} does not exist, skipping.")
            continue

        orig, aug = process_folder(in_dir, out_dir, subdir)
        total_orig += orig
        total_aug += aug

    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Original images : {total_orig}")
    print(f"Augmented images: {total_aug}  (expected: {total_orig * 5})")
    print(f"Total dataset   : {total_orig + total_aug}  (6x = {total_orig * 6})")

    if total_aug == total_orig * 5:
        print("SUCCESS: 5 augmented variants per image, dataset is 6x original size.")
    else:
        print(f"WARNING: Count mismatch. Check errors above.")


if __name__ == "__main__":
    main()
