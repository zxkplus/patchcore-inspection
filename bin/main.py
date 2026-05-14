#!/usr/bin/env python3
"""
Simple PatchCore inference script for industrial inspection.

This script trains a PatchCore model on normal images from one folder
and performs anomaly detection on test images from another folder.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)


class SimpleImageDataset(torch.utils.data.Dataset):
    """Simple dataset to load images from a folder."""
    
    def __init__(self, image_folder, resize=256, imagesize=224):
        self.image_paths = sorted([str(p) for p in Path(image_folder).glob("*") 
                                  if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        self.resize = resize
        self.imagesize = imagesize
        
        # Image transformations
        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.transform_std = [0.229, 0.224, 0.225]
        self.transform_mean = [0.485, 0.456, 0.406]
        
        LOGGER.info(f"Found {len(self.image_paths)} images in {image_folder}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform_img(image)
        return {"image": image}


def create_patchcore_model(
    backbone_name="wideresnet50",
    layers_to_extract_from=["layer2", "layer3"],
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    input_shape=(3, 224, 224),
    pretrain_embed_dimension=1024,
    target_embed_dimension=1024,
    patchsize=3,
    percentage=0.1,
    anomaly_scorer_num_nn=1,
    faiss_on_gpu=True,
    faiss_num_workers=8,
):
    """Create and configure a PatchCore model."""
    # Load backbone
    backbone = patchcore.backbones.load(backbone_name)
    backbone.name = backbone_name

    # Create sampler
    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    # Create nearest neighbor method
    nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

    # Create PatchCore instance
    patchcore_instance = patchcore.patchcore.PatchCore(device)
    patchcore_instance.load(
        backbone=backbone,
        layers_to_extract_from=layers_to_extract_from,
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=patchsize,
        featuresampler=sampler,
        anomaly_scorer_num_nn=anomaly_scorer_num_nn,
        nn_method=nn_method,
    )

    return patchcore_instance


def visualize_results(test_image_paths, segmentations, scores, output_dir):
    """Visualize and save the anomaly detection results."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image_path in enumerate(test_image_paths):
        # Load original image
        original_img = cv2.imread(str(image_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Get segmentation mask
        seg_mask = segmentations[i]
        # Resize segmentation to original image size
        seg_mask_resized = cv2.resize(seg_mask, (original_img.shape[1], original_img.shape[0]))
        
        # Create overlay
        overlay = original_img.copy()
        # Normalize segmentation mask to 0-255 range
        seg_normalized = ((seg_mask_resized - seg_mask_resized.min()) / 
                         (seg_mask_resized.max() - seg_mask_resized.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(seg_normalized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(seg_mask_resized, cmap='hot')
        axes[1].set_title(f'Anomaly Score: {scores[i]:.3f}')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Save figure
        output_path = os.path.join(output_dir, f"result_{Path(image_path).stem}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        LOGGER.info(f"Saved result for {Path(image_path).name} with score {scores[i]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="PatchCore Anomaly Detection on Custom Images")
    parser.add_argument("normal_images_folder", type=str, 
                        help="Path to folder containing normal/good images (512x512)")
    parser.add_argument("test_images_folder", type=str, 
                        help="Path to folder containing test images to detect anomalies")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Directory to save results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (use -1 for CPU)")
    parser.add_argument("--resize", type=int, default=256, 
                        help="Resize dimension for preprocessing")
    parser.add_argument("--imagesize", type=int, default=224, 
                        help="Final input image size for the model")
    parser.add_argument("--percentage", type=float, default=0.1,
                        help="Percentage of features to keep in coreset sampling")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    LOGGER.info(f"Command line arguments: {' '.join(sys.argv)}")
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    LOGGER.info(f"Using device: {device}")
    
    # Create datasets
    LOGGER.info(f"Loading normal images from: {args.normal_images_folder}")
    normal_dataset = SimpleImageDataset(
        args.normal_images_folder, 
        resize=args.resize, 
        imagesize=args.imagesize
    )
    
    LOGGER.info(f"Loading test images from: {args.test_images_folder}")
    test_dataset = SimpleImageDataset(
        args.test_images_folder, 
        resize=args.resize, 
        imagesize=args.imagesize
    )
    
    if len(normal_dataset) == 0:
        LOGGER.error(f"No normal images found in {args.normal_images_folder}")
        sys.exit(1)
    
    if len(test_dataset) == 0:
        LOGGER.error(f"No test images found in {args.test_images_folder}")
        sys.exit(1)
    
    # Create dataloaders
    normal_dataloader = torch.utils.data.DataLoader(
        normal_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create PatchCore model
    LOGGER.info("Creating PatchCore model...")
    patchcore_model = create_patchcore_model(
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
        percentage=args.percentage,
        faiss_on_gpu=(device.type == "cuda"),
    )
    
    # Train model on normal images
    LOGGER.info("Training PatchCore model on normal images...")
    patchcore_model.fit(normal_dataloader)
    LOGGER.info("Model training completed!")
    
    # Perform inference on test images
    LOGGER.info("Performing inference on test images...")
    scores, segmentations, _, _ = patchcore_model.predict(test_dataloader)
    
    # Get test image paths
    test_image_paths = test_dataset.image_paths
    
    # Visualize and save results
    LOGGER.info("Visualizing results...")
    visualize_results(test_image_paths, segmentations, scores, args.output_dir)
    
    # Print summary
    LOGGER.info(f"\nInference completed!")
    LOGGER.info(f"Processed {len(test_image_paths)} test images")
    LOGGER.info(f"Average anomaly score: {np.mean(scores):.3f}")
    LOGGER.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()