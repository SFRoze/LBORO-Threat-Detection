"""
Threat Detection System - Core Detection Module

This module implements the main ThreatDetector class for identifying people
and potential security threats using deep learning models.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import yaml
import time
from datetime import datetime

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Please run: pip install -r requirements.txt")
    raise

from ..utils.logger import setup_logger
from ..utils.alerts import AlertManager


class ThreatDetector:
    """
    Main class for threat detection using deep learning models.
    
    Supports:
    - Person detection
    - Vehicle detection
    - Custom threat object detection
    - Real-time video processing
    - Alert generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the threat detection system.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config['logging'])
        self.alert_manager = AlertManager(self.config['alerts'])
        
        # Initialize model
        self.model = None
        self.custom_model = None
        self._load_models()
        
        # Detection statistics
        self.stats = {
            'total_detections': 0,
            'person_detections': 0,
            'threat_detections': 0,
            'processing_time': 0.0
        }
        
        self.logger.info("ThreatDetector initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Get the directory of this file, then navigate to config
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config" / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration if config file not found."""
        return {
            'model': {
                'primary_model': 'yolov8n.pt',
                'device': 'auto',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.45
            },
            'classes': {
                'person': 0,
                'high_priority': [0]
            },
            'alerts': {
                'person_detection': True,
                'console_output': True,
                'confidence_minimum': 0.7
            },
            'processing': {
                'input_size': [640, 640],
                'draw_boxes': True,
                'draw_labels': True
            },
            'logging': {
                'level': 'INFO',
                'console': True
            }
        }
    
    def _load_models(self):
        """Load YOLO models for detection."""
        try:
            model_name = self.config['model']['primary_model']
            self.logger.info(f"Loading primary model: {model_name}")
            
            # YOLO will automatically download the model if it doesn't exist
            self.model = YOLO(model_name)
            
            # Set device
            device = self.config['model']['device']
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model.to(device)
            self.logger.info(f"Model loaded on device: {device}")
            
            # Load custom model if specified
            custom_model_path = self.config['model'].get('custom_model')
            if custom_model_path and os.path.exists(custom_model_path):
                self.custom_model = YOLO(custom_model_path)
                self.custom_model.to(device)
                self.logger.info(f"Custom model loaded: {custom_model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def detect_image(self, image_path: str, save_output: bool = True) -> Dict:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to the input image
            save_output: Whether to save the annotated output image
            
        Returns:
            Dictionary containing detection results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        start_time = time.time()
        
        # Run inference
        results = self.model(
            image_path,
            conf=self.config['model']['confidence_threshold'],
            iou=self.config['model']['nms_threshold']
        )
        
        # Process results
        detections = self._process_results(results[0], image_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        detections['processing_time'] = processing_time
        
        # Update statistics
        self._update_stats(detections, processing_time)
        
        # Generate alerts if needed
        self.alert_manager.process_detections(detections)
        
        # Save annotated image if requested
        if save_output and self.config['processing']['draw_boxes']:
            self._save_annotated_image(results[0], image_path)
        
        self.logger.info(f"Processed {image_path} in {processing_time:.2f}s - "
                        f"Found {len(detections['objects'])} objects")
        
        return detections
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Detect objects in a video file.
        
        Args:
            video_path: Path to the input video
            output_path: Path for output video (optional)
            
        Returns:
            Dictionary containing aggregated detection results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        aggregated_results = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'detections_by_frame': [],
            'summary': {
                'total_persons': 0,
                'max_persons_per_frame': 0,
                'threat_alerts': 0
            }
        }
        
        frame_count = 0
        skip_frames = self.config['processing'].get('skip_frames', 0)
        
        self.logger.info(f"Processing video: {video_path} ({total_frames} frames)")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Run detection on frame
                results = self.model(
                    frame,
                    conf=self.config['model']['confidence_threshold'],
                    iou=self.config['model']['nms_threshold']
                )
                
                # Process results
                frame_detections = self._process_results(
                    results[0], 
                    f"frame_{frame_count}"
                )
                frame_detections['frame_number'] = frame_count
                
                # Update aggregated results
                aggregated_results['detections_by_frame'].append(frame_detections)
                aggregated_results['processed_frames'] += 1
                
                # Update summary statistics
                person_count = sum(1 for obj in frame_detections['objects'] 
                                 if obj['class'] == 'person')
                aggregated_results['summary']['total_persons'] += person_count
                aggregated_results['summary']['max_persons_per_frame'] = max(
                    aggregated_results['summary']['max_persons_per_frame'],
                    person_count
                )
                
                # Generate alerts
                self.alert_manager.process_detections(frame_detections)
                
                # Draw annotations and save frame
                if out and self.config['processing']['draw_boxes']:
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                
                frame_count += 1
                
                # Progress logging
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if out:
                out.release()
        
        self.logger.info(f"Video processing completed: {frame_count} frames processed")
        return aggregated_results
    
    def process_video_stream(self, source: Union[int, str] = 0) -> None:
        """
        Process real-time video stream (camera or RTSP).
        
        Args:
            source: Video source (camera index or RTSP URL)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        self.logger.info(f"Starting real-time processing from source: {source}")
        self.logger.info("Press 'q' to quit")
        
        fps_limit = self.config['processing'].get('fps_limit', 30)
        frame_time = 1.0 / fps_limit if fps_limit > 0 else 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from video source")
                    break
                
                # Run detection
                results = self.model(
                    frame,
                    conf=self.config['model']['confidence_threshold'],
                    iou=self.config['model']['nms_threshold']
                )
                
                # Process and display results
                detections = self._process_results(results[0], "live_stream")
                
                # Generate alerts
                self.alert_manager.process_detections(detections)
                
                # Draw annotations
                if self.config['processing']['draw_boxes']:
                    annotated_frame = results[0].plot()
                    cv2.imshow('Threat Detection System', annotated_frame)
                
                # Control frame rate
                elapsed = time.time() - start_time
                if frame_time > elapsed:
                    time.sleep(frame_time - elapsed)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Stream processing interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _process_results(self, result, source: str) -> Dict:
        """
        Process YOLO detection results into standardized format.
        
        Args:
            result: YOLO result object
            source: Source identifier (file path, frame number, etc.)
            
        Returns:
            Processed detection dictionary
        """
        detections = {
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'objects': [],
            'counts': {},
            'alerts': []
        }
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i, box in enumerate(boxes.xyxy):
                # Extract box coordinates
                x1, y1, x2, y2 = box.cpu().numpy()
                
                # Get class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                class_name = result.names[class_id]
                
                # Create detection object
                detection = {
                    'class_id': class_id,
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                detections['objects'].append(detection)
                
                # Update counts
                if class_name in detections['counts']:
                    detections['counts'][class_name] += 1
                else:
                    detections['counts'][class_name] = 1
        
        return detections
    
    def _save_annotated_image(self, result, image_path: str):
        """Save annotated image with detection boxes."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.config.get('paths', {}).get('output', 'output'))
            output_dir.mkdir(exist_ok=True)
            
            # Generate output filename
            input_filename = Path(image_path).stem
            output_path = output_dir / f"{input_filename}_detected.jpg"
            
            # Get annotated image and save
            annotated_image = result.plot()
            cv2.imwrite(str(output_path), annotated_image)
            
            self.logger.debug(f"Saved annotated image: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save annotated image: {e}")
    
    def _update_stats(self, detections: Dict, processing_time: float):
        """Update detection statistics."""
        self.stats['total_detections'] += 1
        self.stats['processing_time'] += processing_time
        
        # Count person detections
        person_count = detections['counts'].get('person', 0)
        self.stats['person_detections'] += person_count
        
        # Count threat detections (custom logic based on your threat classes)
        threat_count = 0
        for obj in detections['objects']:
            if obj['class'] in ['weapon', 'suspicious_package']:  # Custom threat classes
                threat_count += 1
        self.stats['threat_detections'] += threat_count
    
    def get_statistics(self) -> Dict:
        """Get current detection statistics."""
        stats = self.stats.copy()
        if stats['total_detections'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_detections']
        else:
            stats['avg_processing_time'] = 0.0
        return stats
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.stats = {
            'total_detections': 0,
            'person_detections': 0,
            'threat_detections': 0,
            'processing_time': 0.0
        }
        self.logger.info("Statistics reset")


if __name__ == "__main__":
    # Example usage
    detector = ThreatDetector()
    
    # Test with a sample image (you would replace this with your actual image)
    sample_image = "data/samples/test_image.jpg"
    if os.path.exists(sample_image):
        results = detector.detect_image(sample_image)
        print("Detection Results:")
        print(f"Found {len(results['objects'])} objects")
        for obj in results['objects']:
            print(f"- {obj['class']}: {obj['confidence']:.2f}")
    else:
        print("No sample image found. Place a test image at 'data/samples/test_image.jpg' to test.")
