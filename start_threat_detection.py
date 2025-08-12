#!/usr/bin/env python3
"""
Threat Detection System - Easy Launcher

This script provides a simple way to start the threat detection system
with your webcam for real-time crime detection.
"""

import sys
import os
from pathlib import Path
import subprocess
import time

def main():
    """Main launcher function."""
    print("🚨" * 20)
    print("   THREAT DETECTION SYSTEM")
    print("   Real-Time Crime Detection")
    print("🚨" * 20)
    
    print("\n🔧 System designed for:")
    print("   ✅ Person detection")
    print("   ✅ Weapon detection (guns, knives, scissors, baseball bats)")
    print("   ✅ Crime detection (person + weapon)")
    print("   ✅ Real-time webcam processing")
    print("   ✅ Visual and audio alerts")
    
    print("\n🎮 Controls:")
    print("   Q or ESC - Quit the system (EASY EXIT!)")
    print("   S - Save screenshot")
    print("   R - Reset statistics")
    
    print("\n⚠️  IMPORTANT:")
    print("   - Make sure your webcam is connected")
    print("   - Good lighting helps detection accuracy")
    print("   - System uses your GPU if available")
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("\n❌ Error: Please run this from the threat_detection_system directory")
        input("Press Enter to exit...")
        return
    
    # Ask user for camera selection
    print("\n📷 Camera Selection:")
    print("   0 - Default webcam (most common)")
    print("   1 - Secondary camera")
    print("   2 - USB camera")
    
    while True:
        try:
            camera_choice = input("\nEnter camera number (0-2, or press Enter for default): ").strip()
            if camera_choice == "":
                camera_id = 0
            else:
                camera_id = int(camera_choice)
                if not 0 <= camera_id <= 2:
                    raise ValueError
            break
        except ValueError:
            print("❌ Please enter a number between 0-2")
    
    # Ask for sensitivity
    print("\n🎯 Detection Sensitivity:")
    print("   1 - High sensitivity (detects more, may have false positives)")
    print("   2 - Medium sensitivity (balanced - recommended)")
    print("   3 - Low sensitivity (only very confident detections)")
    
    while True:
        try:
            sensitivity_choice = input("\nEnter sensitivity (1-3, or press Enter for medium): ").strip()
            if sensitivity_choice == "":
                confidence = 0.5
            elif sensitivity_choice == "1":
                confidence = 0.3
            elif sensitivity_choice == "2":
                confidence = 0.5
            elif sensitivity_choice == "3":
                confidence = 0.7
            else:
                raise ValueError
            break
        except ValueError:
            print("❌ Please enter 1, 2, or 3")
    
    print(f"\n🚀 Starting threat detection with:")
    print(f"   Camera: {camera_id}")
    print(f"   Sensitivity: {'High' if confidence == 0.3 else 'Medium' if confidence == 0.5 else 'Low'}")
    print(f"   Confidence threshold: {confidence}")
    
    print("\n⏳ Initializing system...")
    
    # Add a countdown
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...", end="\r")
        time.sleep(1)
    
    print("   🎬 SYSTEM ACTIVE!     ")
    
    try:
        # Run the real-time threat detector
        cmd = [
            sys.executable,
            "src/realtime_threat_detector.py",
            "--camera", str(camera_id),
            "--confidence", str(confidence)
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  System stopped by user")
    except Exception as e:
        print(f"\n❌ Error running threat detection: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run: python quick_start.py")
        print("2. Check that your webcam is working")
        print("3. Ensure all dependencies are installed")
    
    print("\n👋 Thanks for using the Threat Detection System!")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
