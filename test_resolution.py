def create_patch_map(self, img_shape: Tuple[int, int], 
                        mse_2p: np.ndarray, mse_4p: np.ndarray) -> np.ndarray:
        """
        Create hierarchical patch map based on MSE thresholds.
        
        Args:
            img_shape: (H, W) of original image
            mse_2p: MSE map for 2P×2P patches (32×32)
            mse_4p: MSE map for 4P×4P patches (64×64)
            
        Returns:
            Patch map where 0=16×16, 1=32×32, 2=64×64
        """
        H, W = img_shape
        patch_map = np.zeros((H, W), dtype=np.int32)
        
        # Count patches selected
        count_4p = 0
        count_2p = 0
        
        # Mark 4P×4P patches (64×64)
        h_4p, w_4p = mse_4p.shape
        for i in range(h_4p):
            for j in range(w_4p):
                if mse_4p[i, j] < self.threshold_4:
                    y_start = i * (4 * self.P)
                    x_start = j * (4 * self.P)
                    y_end = min(y_start + 4 * self.P, H)
                    x_end = min(x_start + 4 * self.P, W)
                    patch_map[y_start:y_end, x_start:x_end] = 2
                    count_4p += 1
        
        # Mark 2P×2P patches (32×32) - only where not already marked as 4P×4P
        h_2p, w_2p = mse_2p.shape
        for i in range(h_2p):
            for j in range(w_2p):
                if mse_2p[i, j] < self.threshold_2:
                    y_start = i * (2 * self.P)
                    x_start = j * (2 * self.P)
                    y_end = min(y_start + 2 * self.P, H)
                    x_end = min(x_start + 2import sys
sys.path.append("../")

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import PIL.Image as Image
import numpy as np
import cv2
from typing import Tuple, Dict, List
import random
from datetime import datetime

# Try to import mediapy for better image saving
try:
    import mediapy as media
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False
    print("mediapy not found. Using cv2 for saving images.")
    print("Install with: pip install mediapy")
    print()

# Import your custom transform if available, otherwise use standard
try:
    from src.data.transforms import transforms_imagenet_eval
except:
    def transforms_imagenet_eval(img_size):
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

class HierarchicalPatchAnalyzer:
    def __init__(self, patch_size: int = 32, threshold_2: float = 0.01, threshold_4: float = 0.005):
        """
        Initialize the hierarchical patch analyzer.
        
        Args:
            patch_size: Base patch size P
            threshold_2: MSE threshold for 2P×2P patches
            threshold_4: MSE threshold for 4P×4P patches
        """
        self.P = patch_size
        self.threshold_2 = threshold_2
        self.threshold_4 = threshold_4
        
    def calculate_patch_mse(self, img1: torch.Tensor, img2: torch.Tensor, patch_size: int) -> np.ndarray:
        """
        Calculate MSE for each patch of given size.
        
        Args:
            img1, img2: Images to compare [1, C, H, W]
            patch_size: Size of patches to compute MSE over
            
        Returns:
            MSE map with shape [H//patch_size, W//patch_size]
        """
        assert img1.shape == img2.shape
        _, C, H, W = img1.shape
        
        # Calculate pixel-wise MSE
        mse_pixelwise = F.mse_loss(img1, img2, reduction='none')  # [1, C, H, W]
        
        # Average over channels
        mse_pixelwise = mse_pixelwise.mean(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # Use average pooling to get patch-wise MSE
        mse_patches = F.avg_pool2d(mse_pixelwise, kernel_size=patch_size, stride=patch_size)
        
        return mse_patches.squeeze().cpu().numpy()
    
    def downsample_upsample(self, img: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Downsample and then upsample back to original size.
        
        Args:
            img: Input image [1, C, H, W]
            scale: Downsampling factor (0.5 for 1/2, 0.25 for 1/4)
            
        Returns:
            Reconstructed image at original size
        """
        # Downsample
        img_down = F.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)
        # Upsample back
        img_up = F.interpolate(img_down, scale_factor=1/scale, mode='bilinear', align_corners=False)
        return img_up
    
    def create_patch_map(self, img_shape: Tuple[int, int], 
                        mse_2p: np.ndarray, mse_4p: np.ndarray) -> np.ndarray:
        """
        Create hierarchical patch map based on MSE thresholds.
        
        Args:
            img_shape: (H, W) of original image
            mse_2p: MSE map for 2P×2P patches (32×32)
            mse_4p: MSE map for 4P×4P patches (64×64)
            
        Returns:
            Patch map where 0=16×16, 1=32×32, 2=64×64
        """
        H, W = img_shape
        patch_map = np.zeros((H, W), dtype=np.int32)
        
        # Count patches selected
        count_4p = 0
        count_2p = 0
        
        # Mark 4P×4P patches (64×64)
        h_4p, w_4p = mse_4p.shape
        for i in range(h_4p):
            for j in range(w_4p):
                if mse_4p[i, j] < self.threshold_4:
                    y_start = i * (4 * self.P)
                    x_start = j * (4 * self.P)
                    y_end = min(y_start + 4 * self.P, H)
                    x_end = min(x_start + 4 * self.P, W)
                    patch_map[y_start:y_end, x_start:x_end] = 2
                    count_4p += 1
        
        # Mark 2P×2P patches (32×32) - only where not already marked as 4P×4P
        h_2p, w_2p = mse_2p.shape
        for i in range(h_2p):
            for j in range(w_2p):
                if mse_2p[i, j] < self.threshold_2:
                    y_start = i * (2 * self.P)
                    x_start = j * (2 * self.P)
                    y_end = min(y_start + 2 * self.P, H)
                    x_end = min(x_start + 2 * self.P, W)
                    
                    # Check if this region overlaps with any 4P×4P patch
                    if not np.any(patch_map[y_start:y_end, x_start:x_end] == 2):
                        patch_map[y_start:y_end, x_start:x_end] = 1
                        count_2p += 1
        
        # Everything else remains 0 (16×16 patches)
        print(f"Selected {count_4p} 64×64 patches, {count_2p} 32×32 patches")
        return patch_map
    
    def draw_grid(self, img: np.ndarray, grid_size: int, color: Tuple[int, int, int], thickness: int = 1) -> np.ndarray:
        """
        Draw a grid on the image.
        
        Args:
            img: Image to draw on [H, W, 3]
            grid_size: Size of grid cells
            color: BGR color for grid lines
            thickness: Line thickness
            
        Returns:
            Image with grid
        """
        img_grid = img.copy()
        H, W = img.shape[:2]
        
        # Draw vertical lines
        for x in range(0, W + 1, grid_size):
            cv2.line(img_grid, (x, 0), (x, H), color, thickness)
        
        # Draw horizontal lines
        for y in range(0, H + 1, grid_size):
            cv2.line(img_grid, (0, y), (W, y), color, thickness)
            
        return img_grid
    
    def draw_hierarchical_patches(self, img: np.ndarray, patch_map: np.ndarray) -> np.ndarray:
        """
        Draw hierarchical patch boundaries on image.
        
        Args:
            img: Image to draw on [H, W, 3]
            patch_map: Patch map indicating patch sizes
            
        Returns:
            Image with drawn patch boundaries
        """
        img_vis = img.copy()
        H, W = patch_map.shape
        
        # Draw all patches in white with different thicknesses
        white = (255, 255, 255)
        
        # First, draw the finest grid (P×P) where needed
        for i in range(0, H, self.P):
            for j in range(0, W, self.P):
                if i < H and j < W and patch_map[i, j] == 0:
                    cv2.rectangle(img_vis,
                                (j, i),
                                (min(j + self.P, W-1), min(i + self.P, H-1)),
                                white, 1)
        
        # Draw 2P×2P patches (32×32)
        for i in range(0, H, 2 * self.P):
            for j in range(0, W, 2 * self.P):
                if i < H and j < W and patch_map[i, j] == 1:
                    cv2.rectangle(img_vis,
                                (j, i),
                                (min(j + 2 * self.P, W-1), min(i + 2 * self.P, H-1)),
                                white, 2)
        
        # Draw 4P×4P patches (64×64)
        for i in range(0, H, 4 * self.P):
            for j in range(0, W, 4 * self.P):
                if i < H and j < W and patch_map[i, j] == 2:
                    cv2.rectangle(img_vis, 
                                (j, i), 
                                (min(j + 4 * self.P, W-1), min(i + 4 * self.P, H-1)),
                                white, 3)
        
        return img_vis
    
    def create_mse_heatmap(self, mse_map: np.ndarray, patch_size: int, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a heatmap visualization of MSE values.
        
        Args:
            mse_map: MSE values for each patch
            patch_size: Size of each patch
            img_shape: (H, W) of target image size
            
        Returns:
            Heatmap image [H, W, 3] in BGR format
        """
        H, W = img_shape
        
        # Resize MSE map to match image size
        mse_resized = cv2.resize(mse_map, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Normalize to 0-255 range
        mse_norm = np.clip(mse_resized / mse_resized.max() * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(mse_norm, cv2.COLORMAP_JET)
        
        # Draw grid to show patch boundaries
        heatmap = self.draw_grid(heatmap, patch_size, (255, 255, 255), 1)
        
        return heatmap
    
    def add_text_overlay(self, img: np.ndarray, text: str, position: str = 'top') -> np.ndarray:
        """
        Add text overlay to image.
        
        Args:
            img: Image to add text to
            text: Text to add
            position: 'top' or 'bottom'
            
        Returns:
            Image with text overlay
        """
        img_with_text = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)
        
        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Calculate position
        if position == 'top':
            text_y = 30
        else:
            text_y = img.shape[0] - 10
        
        text_x = (img.shape[1] - text_size[0]) // 2
        
        # Add background for better readability
        cv2.rectangle(img_with_text, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(img_with_text, text, (text_x, text_y), 
                   font, font_scale, color, thickness)
        
        return img_with_text
    
    def process_image(self, img_tensor: torch.Tensor) -> Dict:
        """
        Process image through the hierarchical patch analysis.
        
        Args:
            img_tensor: Input image tensor [1, C, H, W]
            
        Returns:
            Dictionary with all results
        """
        # Step 1: Downsample to 1/2 and upsample back
        img_half_up = self.downsample_upsample(img_tensor, 0.5)
        mse_2p = self.calculate_patch_mse(img_tensor, img_half_up, 2 * self.P)
        
        # Step 2: Downsample to 1/4 and upsample back
        img_quarter_up = self.downsample_upsample(img_tensor, 0.25)
        mse_4p = self.calculate_patch_mse(img_tensor, img_quarter_up, 4 * self.P)
        
        # Step 3: Create hierarchical patch map
        _, _, H, W = img_tensor.shape
        patch_map = self.create_patch_map((H, W), mse_2p, mse_4p)
        
        return {
            'img_half_up': img_half_up,
            'img_quarter_up': img_quarter_up,
            'mse_2p': mse_2p,
            'mse_4p': mse_4p,
            'patch_map': patch_map
        }
    
    def save_image(self, img: np.ndarray, filename: str, output_dir: str = "output"):
        """
        Save image using mediapy or cv2.
        
        Args:
            img: Image in BGR format for cv2
            filename: Name of the file
            output_dir: Directory to save images
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        if HAS_MEDIAPY:
            # Convert BGR to RGB for mediapy
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            media.write_image(filepath, img_rgb)
        else:
            cv2.imwrite(filepath, img)
        
        print(f"Saved: {filepath}")
    
    def visualize_results_cv2(self, img_tensor: torch.Tensor, results: Dict):
        """
        Visualize all intermediate steps and results using cv2 and save to files.
        
        Args:
            img_tensor: Original image tensor
            results: Processing results
        """
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output_patches_{timestamp}"
        
        # Convert tensors to numpy arrays (BGR for cv2)
        def tensor_to_bgr(tensor):
            img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_rgb = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Original image
        img_bgr = tensor_to_bgr(img_tensor)
        img_original = self.add_text_overlay(img_bgr.copy(), "Original Image")
        
        # 1/2 downsampled-upsampled
        img_half_bgr = tensor_to_bgr(results['img_half_up'])
        img_half = self.add_text_overlay(img_half_bgr, "1/2 Down-Up Sampled")
        
        # 1/4 downsampled-upsampled
        img_quarter_bgr = tensor_to_bgr(results['img_quarter_up'])
        img_quarter = self.add_text_overlay(img_quarter_bgr, "1/4 Down-Up Sampled")
        
        # MSE heatmaps
        H, W = img_bgr.shape[:2]
        mse_2p_heatmap = self.create_mse_heatmap(results['mse_2p'], 2 * self.P, (H, W))
        mse_2p_heatmap = self.add_text_overlay(mse_2p_heatmap, 
                                               f"MSE Map {2*self.P}×{2*self.P} (threshold={self.threshold_2:.4f})")
        
        mse_4p_heatmap = self.create_mse_heatmap(results['mse_4p'], 4 * self.P, (H, W))
        mse_4p_heatmap = self.add_text_overlay(mse_4p_heatmap, 
                                               f"MSE Map {4*self.P}×{4*self.P} (threshold={self.threshold_4:.4f})")
        
        # Hierarchical patches visualization
        img_patches = self.draw_hierarchical_patches(img_bgr.copy(), results['patch_map'])
        img_patches_labeled = self.add_text_overlay(img_patches.copy(), 
                                           f"Hierarchical Patches ({4*self.P}×{4*self.P}, {2*self.P}×{2*self.P}, {self.P}×{self.P})")
        
        # Create a grid layout
        row1 = np.hstack([img_original, img_half, img_quarter])
        row2 = np.hstack([mse_2p_heatmap, mse_4p_heatmap, img_patches_labeled])
        final_display = np.vstack([row1, row2])
        
        # Save all images
        print(f"\nSaving images to {output_dir}/")
        
        # Save individual images
        self.save_image(img_original, "01_original.png", output_dir)
        self.save_image(img_half, "02_half_downup.png", output_dir)
        self.save_image(img_quarter, "03_quarter_downup.png", output_dir)
        self.save_image(mse_2p_heatmap, "04_mse_2p_heatmap.png", output_dir)
        self.save_image(mse_4p_heatmap, "05_mse_4p_heatmap.png", output_dir)
        self.save_image(img_patches, "06_patches_clean.png", output_dir)
        self.save_image(img_patches_labeled, "07_patches_labeled.png", output_dir)
        self.save_image(final_display, "08_grid_all.png", output_dir)
        
        # Also save the raw MSE maps as numpy arrays
        np.save(os.path.join(output_dir, "mse_2p_raw.npy"), results['mse_2p'])
        np.save(os.path.join(output_dir, "mse_4p_raw.npy"), results['mse_4p'])
        np.save(os.path.join(output_dir, "patch_map.npy"), results['patch_map'])
        print(f"Saved MSE arrays and patch map as .npy files")
        
        # Print statistics
        patch_map = results['patch_map']
        total_pixels = patch_map.size
        p_pixels = np.sum(patch_map == 0)
        p2_pixels = np.sum(patch_map == 1)
        p4_pixels = np.sum(patch_map == 2)
        
        print(f"\nPatch Statistics:")
        print(f"{self.P}×{self.P} patches: {p_pixels/total_pixels*100:.1f}% of image")
        print(f"{2*self.P}×{2*self.P} patches: {p2_pixels/total_pixels*100:.1f}% of image")
        print(f"{4*self.P}×{4*self.P} patches: {p4_pixels/total_pixels*100:.1f}% of image")
        print(f"\nMSE Statistics:")
        print(f"{2*self.P}×{2*self.P} MSE: min={results['mse_2p'].min():.6f}, max={results['mse_2p'].max():.6f}, mean={results['mse_2p'].mean():.6f}")
        print(f"{4*self.P}×{4*self.P} MSE: min={results['mse_4p'].min():.6f}, max={results['mse_4p'].max():.6f}, mean={results['mse_4p'].mean():.6f}")
        
        # Save statistics to text file
        stats_file = os.path.join(output_dir, "statistics.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Patch Analysis Statistics\n")
            f.write(f"========================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Base patch size (P): {self.P}×{self.P}\n")
            f.write(f"  2P×2P patch size: {2*self.P}×{2*self.P}\n")
            f.write(f"  4P×4P patch size: {4*self.P}×{4*self.P}\n")
            f.write(f"  Threshold for 2P×2P: {self.threshold_2:.6f}\n")
            f.write(f"  Threshold for 4P×4P: {self.threshold_4:.6f}\n\n")
            f.write(f"Patch Distribution:\n")
            f.write(f"  {self.P}×{self.P} patches: {p_pixels/total_pixels*100:.1f}% of image\n")
            f.write(f"  {2*self.P}×{2*self.P} patches: {p2_pixels/total_pixels*100:.1f}% of image\n")
            f.write(f"  {4*self.P}×{4*self.P} patches: {p4_pixels/total_pixels*100:.1f}% of image\n\n")
            f.write(f"MSE Statistics:\n")
            f.write(f"  {2*self.P}×{2*self.P} MSE: min={results['mse_2p'].min():.6f}, max={results['mse_2p'].max():.6f}, mean={results['mse_2p'].mean():.6f}\n")
            f.write(f"  {4*self.P}×{4*self.P} MSE: min={results['mse_4p'].min():.6f}, max={results['mse_4p'].max():.6f}, mean={results['mse_4p'].mean():.6f}\n")
        print(f"Saved statistics to {stats_file}")
        
        print(f"\nAll visualizations saved to {output_dir}/")

def main():
    # Configuration
    split = "val"
    image_size = 512
    patch_size = 16  # Base patch size P (16x16)
    threshold_2 = 0.0002  # Threshold for 32×32 patches (2P×2P)
    threshold_4 = 0.0002  # Threshold for 64×64 patches (4P×4P)
    
    print(f"Configuration:")
    print(f"  Base patch size (P): {patch_size}×{patch_size}")
    print(f"  2P×2P patch size: {2*patch_size}×{2*patch_size}")
    print(f"  4P×4P patch size: {4*patch_size}×{4*patch_size}")
    print(f"  Threshold for both: {threshold_2}")
    print()
    
    # Try to load ImageNet, otherwise create dummy data
    try:
        data_dir = os.path.join("/edrive1/rchoudhu/ILSVRC2012", split)
        val_transform = transforms_imagenet_eval(img_size=image_size)
        data_val = datasets.ImageFolder(
            root=f"{data_dir}",
            transform=val_transform
        )
        
        # Randomly select an image
        random_idx = random.randint(0, len(data_val) - 1)
        print(f"Selected image index: {random_idx}")
        img, label = data_val[random_idx]
        
        # Unnormalize for visualization
        unnorm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_unnorm = unnorm(img)
        
    except:
        print("ImageNet not found. Using random synthetic image with patterns.")
        # Create a synthetic image with some patterns
        img_np = np.zeros((image_size, image_size, 3), dtype=np.float32)
        
        # Add some structured patterns
        for i in range(0, image_size, 64):
            for j in range(0, image_size, 64):
                intensity = (i + j) / (2 * image_size)
                img_np[i:i+32, j:j+32] = [intensity, 0.5, 1-intensity]
                img_np[i+32:i+64, j+32:j+64] = [1-intensity, intensity, 0.5]
        
        # Add some noise
        noise = np.random.randn(image_size, image_size, 3) * 0.1
        img_np = np.clip(img_np + noise, 0, 1)
        
        # Convert to tensor
        img_unnorm = torch.from_numpy(img_np).permute(2, 0, 1).float()
    
    # Prepare image tensor
    img_tensor = img_unnorm.unsqueeze(0)
    print(f"Image shape: {img_tensor.shape}")
    
    # Initialize analyzer
    analyzer = HierarchicalPatchAnalyzer(
        patch_size=patch_size,
        threshold_2=threshold_2,
        threshold_4=threshold_4
    )
    
    # Process image
    print("\nProcessing image...")
    results = analyzer.process_image(img_tensor)
    
    # Visualize results and save to files
    analyzer.visualize_results_cv2(img_tensor, results)

if __name__ == "__main__":
    main()