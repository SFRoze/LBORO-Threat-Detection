#!/usr/bin/env python3
"""
Threat Detection System - Quick Start Script

This script helps you get started with the threat detection system by:
1. Checking system requirements
2. Installing dependencies (if needed)
3. Running a basic test
4. Providing usage examples
"""

import sys
import subprocess
import os
from pathlib import Path
import urllib.request
import time

def main():
    """Main quick start function."""
    print("🚀 Threat Detection System - Quick Start")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check if we're in the right directory
    check_directory()
    
    # Install requirements
    install_requirements()
    
    # Download sample image for testing
    download_sample_image()
    
    # Run basic test
    run_basic_test()
    
    # Show usage examples
    show_usage_examples()


def check_python_version():
    """Check if Python version is compatible."""
    print("\n📋 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        print("Please upgrade Python and try again.")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")


def check_directory():
    """Check if we're in the correct directory structure."""
    print("\n📁 Checking directory structure...")
    
    required_dirs = ['src', 'data', 'output', 'logs']
    required_files = ['README.md', 'requirements.txt', 'src/detect.py']
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"❌ Missing directory: {dir_name}")
            print("Please run this script from the threat_detection_system root directory.")
            sys.exit(1)
    
    for file_name in required_files:
        if not Path(file_name).exists():
            print(f"❌ Missing file: {file_name}")
            sys.exit(1)
    
    print("✅ Directory structure - OK")


def install_requirements():
    """Install Python requirements."""
    print("\n📦 Installing requirements...")
    
    try:
        # Check if requirements are already installed
        import torch
        import cv2
        from ultralytics import YOLO
        print("✅ Core dependencies already installed")
        return
    except ImportError:
        pass
    
    print("Installing dependencies (this may take a few minutes)...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please install manually: pip install -r requirements.txt")
        sys.exit(1)


def download_sample_image():
    """Download a sample image for testing."""
    print("\n🖼️  Downloading sample image for testing...")
    
    sample_path = Path("data/samples/test_image.jpg")
    
    if sample_path.exists():
        print("✅ Sample image already exists")
        return
    
    # Download a sample image from a public source
    sample_url = "https://ultralytics.com/images/bus.jpg"
    
    try:
        print("Downloading sample image...")
        urllib.request.urlretrieve(sample_url, sample_path)
        print("✅ Sample image downloaded")
    except Exception as e:
        print(f"⚠️  Could not download sample image: {e}")
        print("You can manually place a test image at data/samples/test_image.jpg")


def run_basic_test():
    """Run a basic test of the system."""
    print("\n🧪 Running basic system test...")
    
    try:
        # Import the detector
        sys.path.insert(0, str(Path("src")))
        from detection.detector import ThreatDetector
        
        print("Creating detector instance...")
        detector = ThreatDetector()
        
        # Test with sample image if it exists
        sample_path = Path("data/samples/test_image.jpg")
        if sample_path.exists():
            print(f"Testing detection on sample image: {sample_path}")
            start_time = time.time()
            
            results = detector.detect_image(str(sample_path))
            
            processing_time = time.time() - start_time
            
            print(f"✅ Detection completed in {processing_time:.2f} seconds")
            print(f"Found {len(results['objects'])} objects:")
            
            for obj in results['objects'][:5]:  # Show first 5 objects
                print(f"  - {obj['class']}: {obj['confidence']:.3f}")
            
            if len(results['objects']) > 5:
                print(f"  ... and {len(results['objects']) - 5} more")
            
        else:
            print("⚠️  No sample image found, skipping detection test")
            print("✅ System initialization successful")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please check the error messages above and ensure all dependencies are installed.")
        return False
    
    print("✅ Basic system test passed!")
    return True


def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples")
    print("=" * 50)
    
    print("\n1. 📷 Detect objects in an image:")
    print("   python src/detect.py --image data/samples/test_image.jpg")
    
    print("\n2. 🎥 Process a video file:")
    print("   python src/detect.py --video path/to/your/video.mp4 --output output/result.mp4")
    
    print("\n3. 📹 Real-time webcam detection:")
    print("   python src/detect.py --camera 0")
    
    print("\n4. 📊 Run with statistics:")
    print("   python src/detect.py --image data/samples/test_image.jpg --stats")
    
    print("\n5. ⚙️  Use different model (more accurate but slower):")
    print("   python src/detect.py --image data/samples/test_image.jpg --model yolov8s.pt")
    
    print("\n6. 🌐 Start web interface:")
    print("   streamlit run src/web_interface.py")
    
    print("\n7. 📁 Batch process multiple images:")
    print("   python src/detect.py --batch path/to/images/ --output output/")
    
    print("\n📝 Configuration:")
    print("   - Edit src/config/config.yaml to customize detection parameters")
    print("   - Adjust confidence thresholds, alert settings, and more")
    
    print("\n🔧 Model Options:")
    print("   - yolov8n.pt: Fastest, least accurate")
    print("   - yolov8s.pt: Small, balanced")
    print("   - yolov8m.pt: Medium, good accuracy")  
    print("   - yolov8l.pt: Large, high accuracy")
    print("   - yolov8x.pt: Extra large, highest accuracy")
    
    print("\n🚀 Next Steps:")
    print("   1. Try the examples above with your own images/videos")
    print("   2. Customize the configuration in src/config/config.yaml")
    print("   3. Set up alerts (email, webhooks) for your security needs")
    print("   4. Train custom models for specific threat detection")
    
    print("\n🆘 Need Help?")
    print("   - Check README.md for detailed documentation")
    print("   - Run any command with --help for more options")
    print("   - Use the web interface for easier interaction")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
