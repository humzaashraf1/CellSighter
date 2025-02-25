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

def extract_model_input_crops(config_path, output_dir, num_samples=None, random_seed=42):
    """
    Extract and save exact cropped images that are fed into the model,
    with each channel saved as a separate image.
    
    Args:
        config_path (str): Path to the configuration JSON file
        output_dir (str): Directory to save the extracted crops
        num_samples (int, optional): Number of random samples to extract. If None, extract all.
        random_seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load crops
    _, val_crops = load_crops(config["root_dir"],
                              config["channels_path"],
                              config["crop_size"],
                              config["train_set"],
                              config["val_set"],
                              config["to_pad"],
                              blacklist_channels=config["blacklist"] )
    
    # Get crop input size
    crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
    
    # Create dataset
    val_dataset = CellCropsDataset(val_crops, transform=val_transform(crop_input_size), mask=True)
    
    # Load channel names
    channels = []
    with open(config["channels_path"], 'r') as f:
        channels = [line.strip() for line in f if line.strip() not in config["blacklist"]]
    
    # Create a text file with metadata
    with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
        f.write(f"Crop size (initial): {config['crop_size']}\n")
        f.write(f"Crop input size (model): {crop_input_size}\n")
        f.write(f"Number of channels: {len(channels)}\n")
        f.write(f"Total number of crops: {len(val_dataset)}\n")
        f.write("\nChannels used:\n")
        for i, channel in enumerate(channels):
            f.write(f"{i}: {channel}\n")
    
    # Determine which samples to process
    if num_samples is None:
        # Process all samples
        indices = range(len(val_dataset))
    else:
        # Process random samples
        indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    # Process each sample
    for i, idx in enumerate(indices):
        sample = val_dataset[idx]
        
        # Get image data
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
        
        # Save metadata for this crop
        metadata_path = os.path.join(crop_folder_path, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Image ID: {sample['image_id']}\n")
            f.write(f"Cell ID: {sample['cell_id']}\n")
            f.write(f"Label: {sample['label']}\n")
            f.write(f"Shape: {image.shape}\n")  # [channels, height, width]
            f.write(f"Slices: X({sample['slice_x_start']}:{sample['slice_x_end']}), Y({sample['slice_y_start']}:{sample['slice_y_end']})\n")
            f.write("\nChannels:\n")
            for j, channel_name in enumerate(channels):
                if j < image.shape[0]:
                    f.write(f"{j}: {channel_name}\n")
        
        # Save each channel as a separate image
        channels_folder = os.path.join(crop_folder_path, "channels")
        os.makedirs(channels_folder, exist_ok=True)
        
        for j in range(image.shape[0]):
            channel_img = image[j]  # Extract single channel
            
            # Get channel name if available
            channel_name = channels[j] if j < len(channels) else f"channel_{j}"
            
            # Save as TIFF (preserves floating point values)
            channel_path = os.path.join(channels_folder, f"{j:02d}_{channel_name}.tiff")
            tifffile.imwrite(channel_path, channel_img)
        
        # If mask available, save it separately using tifffile
        if mask is not None:
            mask_path = os.path.join(crop_folder_path, "cell_mask.tiff")
            tifffile.imwrite(mask_path, mask)
        
        # Save additional mask types if available
        if 'all_cells_mask' in sample:
            all_cells_mask = sample['all_cells_mask']
            if isinstance(all_cells_mask, torch.Tensor):
                all_cells_mask = all_cells_mask.numpy()
            all_mask_path = os.path.join(crop_folder_path, "all_cells_mask.tiff")
            tifffile.imwrite(all_mask_path, all_cells_mask)
        
        if 'all_cells_mask_seperate' in sample:
            separate_mask = sample['all_cells_mask_seperate']
            if isinstance(separate_mask, torch.Tensor):
                separate_mask = separate_mask.numpy()
            separate_mask_path = os.path.join(crop_folder_path, "labeled_cells_mask.tiff")
            tifffile.imwrite(separate_mask_path, separate_mask)
        
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
