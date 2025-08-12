#!/usr/bin/env python3
"""
Real-Time Threat Detection System

This script provides real-time webcam-based detection of people and weapons
with special "Crime Detected" alerts when weapons are found near people.
"""

import cv2
import numpy as np
import time
import sys
import logging
import threading
import os
from pathlib import Path
from datetime import datetime
import winsound  # For Windows sound alerts
import argparse

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from models.weapon_detector import WeaponDetector
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class RealTimeThreatDetector:
    """
    Real-time threat detection system for webcam feeds.
    
    Features:
    - Person detection
    - Weapon detection
    - Crime scenario detection (person + weapon)
    - Visual and audio alerts
    - Real-time processing
    """
    
    def __init__(self, camera_id=0, confidence_threshold=0.5):
        """
        Initialize the real-time threat detector.
        
        Args:
            camera_id: Camera index (usually 0 for default webcam)
            confidence_threshold: Minimum confidence for detections
        """
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize weapon detector
        self.logger.info("Initializing weapon detector...")
        self.weapon_detector = WeaponDetector()
        
        # Camera setup
        self.cap = None
        
        # Detection statistics
        self.stats = {
            'frames_processed': 0,
            'persons_detected': 0,
            'weapons_detected': 0,
            'crimes_detected': 0,
            'start_time': time.time()
        }
        
        # Alert settings
        self.last_crime_alert = 0
        self.crime_alert_cooldown = 2.0  # seconds
        
        # Display settings
        self.window_name = "🚨 Real-Time Threat Detection System"
        self.display_size = (1280, 720)  # Resize display for better viewing
        
        # Screenshots directory setup
        self.screenshots_dir = Path("screenshots")
        self.setup_screenshots_directory()
        
        self.logger.info("Real-time threat detector initialized")
    
    def setup_screenshots_directory(self):
        """Create screenshots directory if it doesn't exist."""
        try:
            # Create screenshots directory in the project root
            project_root = Path(__file__).parent.parent  # Go up from src/ to project root
            self.screenshots_dir = project_root / "screenshots"
            
            # Create the directory if it doesn't exist
            self.screenshots_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Screenshots directory ready: {self.screenshots_dir}")
            
        except Exception as e:
            self.logger.warning(f"Could not create screenshots directory: {e}")
            # Fallback to current directory
            self.screenshots_dir = Path(".")
    
    def initialize_camera(self):
        """Initialize the camera."""
        self.logger.info(f"Initializing camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {self.camera_id}. Please check if camera is available.")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Camera initialized: {width}x{height} @ {fps:.1f}fps")
        
        return True
    
    def play_alert_sound(self, alert_type="crime"):
        """Play alert sound for different types of detections."""
        try:
            if alert_type == "crime":
                # Play a series of beeps for crime detection
                frequency = 1000  # Hz
                duration = 200    # ms
                for i in range(3):
                    winsound.Beep(frequency, duration)
                    time.sleep(0.1)
            elif alert_type == "weapon":
                # Single beep for weapon detection
                winsound.Beep(800, 500)
        except Exception as e:
            self.logger.warning(f"Could not play alert sound: {e}")
    
    def draw_status_overlay(self, image):
        """Draw status information overlay on the image."""
        overlay = image.copy()
        
        # Calculate stats
        elapsed_time = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / elapsed_time if elapsed_time > 0 else 0
        
        # Status panel background
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Add transparency
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Status text
        status_lines = [
            f"🚨 THREAT DETECTION SYSTEM",
            f"FPS: {fps:.1f} | Frames: {self.stats['frames_processed']}",
            f"Persons: {self.stats['persons_detected']} | Weapons: {self.stats['weapons_detected']}",
            f"CRIMES DETECTED: {self.stats['crimes_detected']}"
        ]
        
        y_offset = 30
        for i, line in enumerate(status_lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)  # Yellow for title, white for others
            if i == 3 and self.stats['crimes_detected'] > 0:
                color = (0, 0, 255)  # Red for crimes
            
            cv2.putText(
                image,
                line,
                (20, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return image
    
    def process_detections(self, results):
        """Process detection results and generate alerts."""
        current_time = time.time()
        
        # Update statistics
        self.stats['persons_detected'] += len(results['persons'])
        self.stats['weapons_detected'] += len(results['weapons'])
        
        # Handle crime detection
        if results['crimes']:
            self.stats['crimes_detected'] += len(results['crimes'])
            
            # Play alert sound if cooldown has passed
            if current_time - self.last_crime_alert >= self.crime_alert_cooldown:
                self.logger.warning(f"🚨 CRIME DETECTED! Found {len(results['crimes'])} potential threats")
                
                # Play alert sound in a separate thread to avoid blocking
                alert_thread = threading.Thread(target=self.play_alert_sound, args=("crime",))
                alert_thread.daemon = True
                alert_thread.start()
                
                self.last_crime_alert = current_time
        
        # Log weapon detections
        if results['weapons']:
            for weapon in results['weapons']:
                self.logger.warning(f"⚠️  Weapon detected: {weapon['class']} (confidence: {weapon['confidence']:.2f})")
    
    def run(self):
        """Run the real-time detection system."""
        try:
            # Initialize camera
            if not self.initialize_camera():
                return
            
            self.logger.info("Starting real-time threat detection...")
            self.logger.info("Press 'q' or ESC to quit, 's' to save screenshot, 'r' to reset statistics")
            
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_size[0], self.display_size[1])
            
            while True:
                start_time = time.time()
                
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                # Run detection
                results = self.weapon_detector.detect_objects(frame, self.confidence_threshold)
                
                # Draw detection results
                annotated_frame = self.weapon_detector.draw_detections(frame, results)
                
                # Process detections for alerts
                self.process_detections(results)
                
                # Draw status overlay
                annotated_frame = self.draw_status_overlay(annotated_frame)
                
                # Display frame
                cv2.imshow(self.window_name, annotated_frame)
                
                # Update statistics
                self.stats['frames_processed'] += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    self.logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    # Save screenshot to screenshots directory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"threat_detection_screenshot_{timestamp}.jpg"
                    filepath = self.screenshots_dir / filename
                    
                    success = cv2.imwrite(str(filepath), annotated_frame)
                    if success:
                        self.logger.info(f"Screenshot saved: {filepath}")
                    else:
                        self.logger.error(f"Failed to save screenshot: {filepath}")
                elif key == ord('r'):
                    # Reset statistics
                    self.stats = {
                        'frames_processed': 0,
                        'persons_detected': 0,
                        'weapons_detected': 0,
                        'crimes_detected': 0,
                        'start_time': time.time()
                    }
                    self.logger.info("Statistics reset")
                
                # Control frame rate (aim for ~30 FPS)
                processing_time = time.time() - start_time
                target_fps = 30
                frame_time = 1.0 / target_fps
                
                if processing_time < frame_time:
                    time.sleep(frame_time - processing_time)
        
        except KeyboardInterrupt:
            self.logger.info("Detection interrupted by user (Ctrl+C)")
        
        except Exception as e:
            self.logger.error(f"Error in detection loop: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed_time = time.time() - self.stats['start_time']
        avg_fps = self.stats['frames_processed'] / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info("=" * 50)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Runtime: {elapsed_time:.1f} seconds")
        self.logger.info(f"Frames processed: {self.stats['frames_processed']}")
        self.logger.info(f"Average FPS: {avg_fps:.1f}")
        self.logger.info(f"Persons detected: {self.stats['persons_detected']}")
        self.logger.info(f"Weapons detected: {self.stats['weapons_detected']}")
        self.logger.info(f"Crimes detected: {self.stats['crimes_detected']}")
        self.logger.info("=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Real-Time Threat Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default webcam (camera 0)
  python src/realtime_threat_detector.py

  # Use specific camera
  python src/realtime_threat_detector.py --camera 1

  # Adjust sensitivity
  python src/realtime_threat_detector.py --confidence 0.3

Controls:
  q or ESC - Quit
  s - Save screenshot
  r - Reset statistics
        """
    )
    
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera index (default: 0 for default webcam)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (0.0-1.0, default: 0.5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate confidence threshold
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Initialize and run detector
    try:
        detector = RealTimeThreatDetector(
            camera_id=args.camera,
            confidence_threshold=args.confidence
        )
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
