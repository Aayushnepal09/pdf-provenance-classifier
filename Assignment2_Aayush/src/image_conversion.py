#!/usr/bin/env python3
"""
Baseline PDF Binary Image Conversion (Adapted for Assignment 2)
"""

import os
import math
from PIL import Image
import numpy as np

def pdf_to_binary_image(pdf_path, output_path, width=None):
    with open(pdf_path, 'rb') as f:
        binary_data = f.read()
    
    byte_array = np.frombuffer(binary_data, dtype=np.uint8)
    total_pixels = len(byte_array)
    
    if width is None:
        width = int(math.sqrt(total_pixels))
    
    height = math.ceil(total_pixels / width)
    
    required_pixels = width * height
    if total_pixels < required_pixels:
        padding = np.zeros(required_pixels - total_pixels, dtype=np.uint8)
        byte_array = np.concatenate([byte_array, padding])
    
    image_array = byte_array[:width * height].reshape(height, width)
    image = Image.fromarray(image_array, mode='L')
    image.save(output_path, 'PNG')
    return image_array.shape

def convert_pdf_directory(input_dir, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    print(f"Converting {len(pdf_files)} PDF files to binary images in {output_dir}...")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(input_dir, pdf_file)
        png_file = f"{prefix}_{pdf_file.replace('.pdf', '.png')}"
        png_path = os.path.join(output_dir, png_file)
        
        try:
            shape = pdf_to_binary_image(pdf_path, png_path)
            print(f"[{i:3d}/{len(pdf_files)}] {pdf_file} -> {png_file}")
        except Exception as e:
            print(f"[{i:3d}/{len(pdf_files)}] ERROR converting {pdf_file}: {e}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'original_pdfs')
    
    # We load source PDFs directly from their original roots and copy to data/original_pdfs/png
    root_dir = os.path.dirname(base_dir) # ForensicsDetective root
    
    output_dir = os.path.join(base_dir, 'data', 'original_pdfs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=== Converting PDFs to centralized original path ===")
    dirs_and_prefixes = [
        ('word_pdfs', 'word'), 
        ('google_docs_pdfs', 'google'), 
        ('python_pdfs', 'python')
    ]
    for d, prefix in dirs_and_prefixes:
        src = os.path.join(root_dir, d)
        if os.path.exists(src):
            convert_pdf_directory(src, output_dir, prefix)
            
    print("\nConversion (Baseline generation) complete!")

if __name__ == "__main__":
    main()