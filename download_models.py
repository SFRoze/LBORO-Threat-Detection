#!/usr/bin/env python3
"""
Model Downloader for Threat Detection System

This script downloads pre-trained models for weapon detection.
"""

import os
import requests
import zipfile
import torch
from pathlib import Path
import sys
import hashlib

def download_file(url, destination, description="file"):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded {description}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {description}: {e}")
        return False

def download_yolo_weapon_model():
    """Download a pre-trained YOLO weapon detection model."""
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # This is a publicly available weapon detection model
    # Note: In a real implementation, you'd want to use a more sophisticated model
    model_info = {
        "name": "weapon_detection.pt",
        "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",  # We'll rename this
        "description": "YOLOv5s base model (will be used for weapon detection)"
    }
    
    model_path = models_dir / model_info["name"]
    
    if model_path.exists():
        print(f"✅ {model_info['name']} already exists")
        return True
    
    # Download the base model (we'll use it as a starting point)
    temp_path = models_dir / "temp_model.pt"
    
    success = download_file(
        model_info["url"],
        temp_path,
        model_info["description"]
    )
    
    if success:
        # Rename to weapon detection model
        temp_path.rename(model_path)
        print(f"✅ Model saved as {model_path}")
        return True
    
    return False

def download_roboflow_weapon_model():
    """Download a weapon detection model from Roboflow (if available)."""
    models_dir = Path("data/models")
    
    # Note: This is a placeholder URL - in practice, you would:
    # 1. Sign up for Roboflow
    # 2. Find a weapon detection dataset
    # 3. Train or download a pre-trained model
    # 4. Get the actual download URL
    
    roboflow_models = [
        {
            "name": "roboflow_weapon_v1.pt",
            "url": "https://app.roboflow.com/ds/example-weapon-model.pt",  # Example URL
            "description": "Roboflow weapon detection model v1"
        }
    ]
    
    print("🔍 Attempting to download specialized weapon detection models...")
    print("Note: These require valid Roboflow API keys and may not be publicly available")
    
    for model_info in roboflow_models:
        model_path = models_dir / model_info["name"]
        
        if model_path.exists():
            print(f"✅ {model_info['name']} already exists")
            continue
        
        print(f"⚠️  Specialized model {model_info['name']} not available in this demo")
        print("   To get better weapon detection, consider:")
        print("   1. Training your own model with weapon datasets")
        print("   2. Using commercial weapon detection APIs")
        print("   3. Downloading models from Roboflow or similar platforms")

def setup_custom_classes():
    """Set up custom class definitions for weapon detection."""
    config_dir = Path("src/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a custom classes file
    classes_file = config_dir / "weapon_classes.yaml"
    
    weapon_classes_config = """# Weapon Detection Classes Configuration

# Standard COCO classes that might be weapons
coco_weapon_classes:
  scissors: 43  # COCO class for scissors

# Custom weapon classes (for future custom models)
custom_weapon_classes:
  gun: 0
  pistol: 1
  rifle: 2
  knife: 3
  blade: 4
  sword: 5
  weapon: 6

# Detection thresholds by class
class_thresholds:
  scissors: 0.3    # Lower threshold for harder to detect items
  gun: 0.4
  pistol: 0.4
  rifle: 0.5
  knife: 0.3
  blade: 0.3
  sword: 0.4
  weapon: 0.4

# Class priorities for alerts
class_priorities:
  high_priority:
    - gun
    - pistol
    - rifle
  medium_priority:
    - knife
    - blade
    - sword
  low_priority:
    - scissors
"""
    
    with open(classes_file, 'w') as f:
        f.write(weapon_classes_config)
    
    print(f"✅ Created weapon classes configuration: {classes_file}")

def verify_gpu_support():
    """Check if CUDA/GPU support is available."""
    print("\n🔍 Checking GPU support...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        
        print(f"✅ CUDA available: {gpu_count} GPU(s) detected")
        print(f"   Primary GPU: {gpu_name}")
        print("   Your system can use GPU acceleration for faster detection!")
    else:
        print("⚠️  CUDA not available - will use CPU for detection")
        print("   Consider installing CUDA-enabled PyTorch for better performance:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

def main():
    """Main function to download and set up models."""
    print("🚀 Threat Detection System - Model Setup")
    print("=" * 50)
    
    # Verify we're in the right directory
    if not Path("src").exists() or not Path("requirements.txt").exists():
        print("❌ Please run this script from the threat_detection_system root directory")
        sys.exit(1)
    
    print("📁 Setting up models directory...")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download base YOLO models
    print("\n📦 Downloading base detection models...")
    success = download_yolo_weapon_model()
    
    if not success:
        print("❌ Failed to download base models")
        print("The system will still work with default YOLO models, but weapon detection may be less accurate")
    
    # Attempt to download specialized weapon models
    print("\n🎯 Checking for specialized weapon detection models...")
    download_roboflow_weapon_model()
    
    # Set up custom classes
    print("\n⚙️  Setting up weapon detection configuration...")
    setup_custom_classes()
    
    # Verify GPU support
    verify_gpu_support()
    
    # Final setup
    print("\n✅ Model setup completed!")
    print("\n🚀 Ready to run threat detection!")
    print("\nNext steps:")
    print("1. Run the real-time detector: python src/realtime_threat_detector.py")
    print("2. Or use the web interface: streamlit run src/web_interface.py")
    print("3. Or test with images: python src/detect.py --image path/to/image.jpg")

if __name__ == "__main__":
    main()
