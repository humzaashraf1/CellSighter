import os
import numpy as np
import tifffile
import torch
import argparse
import json
from pathlib import Path
from data.utils import load_crops
from data.data import CellCropsDataset
from data.transform import val_transform
import cv2

def extract_model_input_crops(config_path, output_dir, num_samples=None, random_seed=42):
    """
    Extract and save cropped images used as model inputs, along with additional outputs:
      - A multipage TIFF combining all channels.
      - An image of the original image with a drawn bounding box for the crop.
      - A tiled image (5x5 grid) showing each channel in grayscale.
      - Optionally, the cropped mask saved as a PNG.

    Args:
        config_path (str): Path to the configuration JSON file.
        output_dir (str): Directory where extracted crops and outputs are saved.
        num_samples (int, optional): Number of random samples to extract. If None, process all.
        random_seed (int): Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load crops and dataset
    _, val_crops = load_crops(config["root_dir"],
                              config["channels_path"],
                              config["crop_size"],
                              config["train_set"],
                              config["val_set"],
                              config["to_pad"],
                              blacklist_channels=config["blacklist"])
    
    crop_input_size = config.get("crop_input_size", 100)
    val_dataset = CellCropsDataset(val_crops, transform=val_transform(crop_input_size), mask=True)
    
    # Load channel names from file (skipping blacklisted ones)
    with open(config["channels_path"], 'r') as f:
        channels = [line.strip() for line in f if line.strip() not in config["blacklist"]]
    
    # Create global metadata file
    meta_global_path = os.path.join(output_dir, "metadata.txt")
    with open(meta_global_path, 'w') as f:
        f.write(f"Crop size (initial): {config['crop_size']}\n")
        f.write(f"Crop input size (model): {crop_input_size}\n")
        f.write(f"Number of channels: {len(channels)}\n")
        f.write(f"Total number of crops: {len(val_dataset)}\n")
        f.write("\nChannels used:\n")
        for i, channel in enumerate(channels):
            f.write(f"{i}: {channel}\n")
    
    # Determine indices to process
    if num_samples is None:
        indices = range(len(val_dataset))
    else:
        indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    # Process each sample
    for i, idx in enumerate(indices):
        sample = val_dataset[idx]
        
        # Get image data (assumed shape: [channels, height, width])
        image = sample['image']
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Get mask if available
        mask = None
        if 'mask' in sample:
            mask = sample['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
        
        # Create folder for this crop
        crop_folder_name = f"crop_image{sample['image_id']}_cell{sample['cell_id']}"
        crop_folder_path = os.path.join(output_dir, crop_folder_name)
        os.makedirs(crop_folder_path, exist_ok=True)
        
        # Save per-crop metadata
        metadata_path = os.path.join(crop_folder_path, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Image ID: {sample['image_id']}\n")
            f.write(f"Cell ID: {sample['cell_id']}\n")
            f.write(f"Label: {sample['label']}\n")
            f.write(f"Shape: {image.shape}\n")
            f.write(f"Slices: X({sample['slice_x_start']}:{sample['slice_x_end']}), Y({sample['slice_y_start']}:{sample['slice_y_end']})\n")
            f.write("\nChannels:\n")
            for j, channel_name in enumerate(channels):
                if j < image.shape[0]:
                    f.write(f"{j}: {channel_name}\n")
        
        # Save each channel as a separate TIFF
        channels_folder = os.path.join(crop_folder_path, "channels")
        os.makedirs(channels_folder, exist_ok=True)
        for j in range(image.shape[0]):
            channel_img = image[j]
            channel_name = channels[j] if j < len(channels) else f"channel_{j}"
            channel_path = os.path.join(channels_folder, f"{j:02d}_{channel_name}.tiff")
            tifffile.imwrite(channel_path, channel_img)
        
        # Save a multipage TIFF combining all channels
        combined_tiff_path = os.path.join(crop_folder_path, "combined_channels.tiff")
        tifffile.imwrite(combined_tiff_path, image)
        
        # Draw bounding box on the original image if available using cv2
        if 'original_image' in sample:
            orig_img = sample['original_image']
            if isinstance(orig_img, torch.Tensor):
                orig_img = orig_img.numpy()
            # If grayscale (2D), convert to BGR; if in (channels, height, width) format, transpose it.
            if orig_img.ndim == 2:
                orig_img_color = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
            elif orig_img.ndim == 3:
                if orig_img.shape[0] <= 4:
                    orig_img_color = np.transpose(orig_img, (1, 2, 0))
                    if orig_img_color.shape[2] == 1:
                        orig_img_color = cv2.cvtColor(orig_img_color, cv2.COLOR_GRAY2BGR)
                else:
                    orig_img_color = orig_img.copy()
            else:
                orig_img_color = orig_img.copy()
            top_left = (sample['slice_x_start'], sample['slice_y_start'])
            bottom_right = (sample['slice_x_end'], sample['slice_y_end'])
            cv2.rectangle(orig_img_color, top_left, bottom_right, (0, 0, 255), thickness=2)
            orig_with_box_path = os.path.join(crop_folder_path, "original_with_bbox.png")
            cv2.imwrite(orig_with_box_path, orig_img_color)
        else:
            print(f"Sample {sample['cell_id']} does not include an 'original_image'; skipping bounding box overlay.")
        
        # Create a tiled image of channels (5x5 grid, up to 25 channels)
        num_channels = image.shape[0]
        grid_rows, grid_cols = 5, 5
        tile_h, tile_w = image.shape[1], image.shape[2]
        tiled_img = np.zeros((grid_rows * tile_h, grid_cols * tile_w), dtype=np.uint8)
        for idx_channel in range(min(num_channels, grid_rows * grid_cols)):
            row = idx_channel // grid_cols
            col = idx_channel % grid_cols
            channel_img = image[idx_channel]
            if channel_img.dtype != np.uint8:
                channel_img = cv2.normalize(channel_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tiled_img[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w] = channel_img
        tiled_img_path = os.path.join(crop_folder_path, "tiled_channels.png")
        cv2.imwrite(tiled_img_path, tiled_img)
        
        # Save the mask as a PNG if available
        if mask is not None:
            # Attempt to squeeze the mask to 2D
            mask_squeezed = np.squeeze(mask)
            if mask_squeezed.ndim != 2:
                if mask_squeezed.size == crop_input_size * crop_input_size:
                    mask_squeezed = mask_squeezed.reshape(crop_input_size, crop_input_size)
                else:
                    print(f"Warning: Unexpected mask shape {mask.shape} for sample {sample['cell_id']}, skipping mask saving.")
                    mask_squeezed = None
            if mask_squeezed is not None:
                if mask_squeezed.dtype != np.uint8:
                    mask_uint8 = cv2.normalize(mask_squeezed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    mask_uint8 = mask_squeezed
                mask_png_path = os.path.join(crop_folder_path, "mask.png")
                cv2.imwrite(mask_png_path, mask_uint8)
        
        if (i + 1) % 10 == 0 or i == len(indices) - 1:
            print(f"Processed {i + 1}/{len(indices)} crops")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract cropped images used as model inputs')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted crops')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of random samples to extract (if None, extract all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    extract_model_input_crops(args.config, args.output_dir, args.num_samples, args.seed)
