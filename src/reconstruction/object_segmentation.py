#!/usr/bin/env python3
"""
Object Segmentation - Phase 3B
Automatic object detection and segmentation using SAM (Segment Anything Model)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    import torch
except ImportError:
    print("❌ SAM dependencies not installed. Install with:")
    print("   pip install segment-anything torch torchvision")
    sys.exit(1)

class ObjectSegmenter:
    def __init__(self, model_type="vit_h", checkpoint_path=None):
        """
        Initialize SAM model for object segmentation
        
        Args:
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            checkpoint_path: Path to SAM checkpoint file
        """
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Initializing SAM on device: {self.device}")
        
        # Try to load SAM model
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            self.predictor = SamPredictor(self.sam)
            print(f"✅ SAM model loaded from: {checkpoint_path}")
        else:
            print("⚠️  SAM checkpoint not found. Will download automatically on first use.")
            print("💡 Download SAM checkpoints from: https://github.com/facebookresearch/segment-anything")
            self.sam = None
            self.mask_generator = None
            self.predictor = None
    
    def download_sam_checkpoint(self):
        """Download SAM checkpoint if not available"""
        checkpoint_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        checkpoint_file = f"sam_{self.model_type}.pth"
        
        if not os.path.exists(checkpoint_file):
            print(f"📥 Downloading SAM checkpoint: {self.model_type}")
            import urllib.request
            try:
                urllib.request.urlretrieve(checkpoint_urls[self.model_type], checkpoint_file)
                print(f"✅ Downloaded: {checkpoint_file}")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                return None
        
        return checkpoint_file
    
    def segment_objects(self, image_path, min_area=500, max_objects=10):
        """
        Segment objects in an image using SAM
        
        Args:
            image_path: Path to input image
            min_area: Minimum area for valid objects (pixels)
            max_objects: Maximum number of objects to segment
            
        Returns:
            List of object masks and metadata
        """
        if self.sam is None:
            checkpoint_file = self.download_sam_checkpoint()
            if checkpoint_file:
                self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint_file)
                self.sam.to(device=self.device)
                self.mask_generator = SamAutomaticMaskGenerator(self.sam)
                self.predictor = SamPredictor(self.sam)
            else:
                print("❌ Could not load SAM model")
                return []
        
        # Load and process image
        print(f"🔍 Segmenting objects in: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        try:
            masks = self.mask_generator.generate(image_rgb)
            print(f"🎯 Found {len(masks)} potential objects")
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")
            return []
        
        # Filter masks by area and quality
        valid_objects = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            area = mask_data['area']
            stability_score = mask_data.get('stability_score', 0)
            
            if area >= min_area and stability_score > 0.7:
                valid_objects.append({
                    'id': len(valid_objects),
                    'mask': mask,
                    'area': area,
                    'stability_score': stability_score,
                    'bbox': mask_data.get('bbox', [0, 0, 0, 0])
                })
                
                if len(valid_objects) >= max_objects:
                    break
        
        print(f"✅ Filtered to {len(valid_objects)} valid objects")
        return valid_objects, image_rgb
    
    def visualize_segmentation(self, image, objects, output_path="segmentation_result.png"):
        """Visualize segmentation results"""
        fig, axes = plt.subplots(1, min(len(objects) + 1, 6), figsize=(20, 4))
        if len(objects) == 0:
            axes = [axes]
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Individual object masks
        for i, obj in enumerate(objects[:5]):  # Show up to 5 objects
            if i + 1 < len(axes):
                # Create colored mask
                colored_mask = np.zeros_like(image)
                color = plt.cm.tab10(i)[:3]  # Get unique color
                colored_mask[obj['mask']] = [int(c * 255) for c in color]
                
                # Overlay on original image
                overlay = image.copy()
                overlay[obj['mask']] = colored_mask[obj['mask']]
                
                axes[i + 1].imshow(overlay)
                axes[i + 1].set_title(f"Object {obj['id']}\nArea: {obj['area']}")
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Segmentation visualization saved: {output_path}")
        plt.close()
    
    def extract_object_regions(self, image, objects, output_dir="segmented_objects"):
        """Extract individual object regions as separate images"""
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_objects = []
        for obj in objects:
            # Create bounding box crop
            x, y, w, h = obj['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Extract object region
            object_image = image[y:y+h, x:x+w].copy()
            object_mask = obj['mask'][y:y+h, x:x+w]
            
            # Apply mask (set background to white)
            object_image[~object_mask] = [255, 255, 255]
            
            # Save extracted object
            object_filename = f"{output_dir}/object_{obj['id']:03d}.png"
            Image.fromarray(object_image).save(object_filename)
            
            extracted_objects.append({
                'id': obj['id'],
                'image_path': object_filename,
                'area': obj['area'],
                'bbox': obj['bbox']
            })
            
            print(f"💾 Extracted object {obj['id']} → {object_filename}")
        
        return extracted_objects

def create_sample_scene():
    """Create a sample RGB image for testing segmentation"""
    print("🎨 Creating sample scene for testing...")
    
    # Create a simple scene with multiple objects
    scene = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Object 1: Red circle
    cv2.circle(scene, (150, 150), 60, (255, 100, 100), -1)
    
    # Object 2: Blue rectangle  
    cv2.rectangle(scene, (300, 100), (450, 200), (100, 100, 255), -1)
    
    # Object 3: Green triangle
    pts = np.array([[500, 300], [450, 400], [550, 400]], np.int32)
    cv2.fillPoly(scene, [pts], (100, 255, 100))
    
    # Object 4: Yellow ellipse
    cv2.ellipse(scene, (200, 350), (80, 40), 0, 0, 360, (255, 255, 100), -1)
    
    # Add some noise/texture
    noise = np.random.randint(-20, 20, scene.shape, dtype=np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    sample_path = "sample_scene.png"
    cv2.imwrite(sample_path, scene)
    print(f"✅ Sample scene created: {sample_path}")
    return sample_path

def main():
    """Main segmentation demonstration"""
    print("============================================================")
    print("🔍 OBJECT SEGMENTATION DEMO - PHASE 3B")
    print("============================================================")
    
    # Check if we have a sample image, create one if not
    image_path = "sample_scene.png"
    if not os.path.exists(image_path):
        image_path = create_sample_scene()
    
    try:
        # Initialize segmenter
        print("\n🤖 Initializing SAM segmenter...")
        segmenter = ObjectSegmenter(model_type="vit_b")  # Use smaller model for demo
        
        # Segment objects
        print("\n🔍 Segmenting objects...")
        objects, original_image = segmenter.segment_objects(image_path, min_area=300, max_objects=8)
        
        if len(objects) == 0:
            print("❌ No objects found in image")
            return
        
        # Visualize results
        print("\n📊 Creating visualization...")
        segmenter.visualize_segmentation(original_image, objects)
        
        # Extract individual objects
        print("\n💾 Extracting individual objects...")
        extracted = segmenter.extract_object_regions(original_image, objects)
        
        # Summary
        print(f"\n✅ Segmentation Complete!")
        print(f"   📊 Objects found: {len(objects)}")
        print(f"   💾 Objects extracted: {len(extracted)}")
        print(f"   📁 Output directory: segmented_objects/")
        print(f"   📊 Visualization: segmentation_result.png")
        
        print("\n🔄 Next Steps:")
        print("   1. Use extract_object_rgbd.py to get depth information")
        print("   2. Run reconstruct_multi_objects.py for 3D reconstruction")
        print("   3. Create multi-object simulation scene")
        
    except Exception as e:
        print(f"❌ Error during segmentation: {e}")
        print("💡 Make sure SAM dependencies are installed:")
        print("   pip install segment-anything torch torchvision opencv-python")

if __name__ == "__main__":
    main() 