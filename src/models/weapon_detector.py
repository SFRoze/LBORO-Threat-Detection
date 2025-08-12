"""
Weapon Detection Model Manager

This module handles downloading and managing pre-trained weapon detection models.
"""

import os
import requests
import zipfile
import torch
from pathlib import Path
import logging
from typing import Optional, List, Dict, Tuple
import numpy as np

try:
    from ultralytics import YOLO
    import cv2
except ImportError as e:
    print(f"Required packages not installed: {e}")
    raise


class WeaponDetector:
    """
    Specialized weapon detection using pre-trained models.
    
    This class combines YOLO person detection with specialized weapon detection
    to identify potential threats and crime scenarios.
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize the weapon detector.
        
        Args:
            model_dir: Directory to store model files
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.person_model = None
        self.weapon_model = None
        
        # Weapon class mappings (these will be updated when we load the model)
        self.weapon_classes = [
            'gun', 'pistol', 'rifle', 'knife', 'sword', 'weapon', 'baseball bat', 'bat', 'stick'
        ]
        
        self._load_models()
    
    def _load_models(self):
        """Load or download the required models."""
        self.logger.info("Loading detection models...")
        
        # Load standard YOLO for person detection
        try:
            self.person_model = YOLO('yolov8n.pt')
            self.logger.info("Person detection model loaded")
        except Exception as e:
            self.logger.error(f"Failed to load person detection model: {e}")
            raise
        
        # Try to load weapon detection model
        weapon_model_path = self.model_dir / "weapon_detection.pt"
        
        if weapon_model_path.exists():
            try:
                self.weapon_model = YOLO(str(weapon_model_path))
                self.logger.info("Weapon detection model loaded from local file")
            except Exception as e:
                self.logger.warning(f"Failed to load local weapon model: {e}")
                self._download_weapon_model()
        else:
            self._download_weapon_model()
    
    def _download_weapon_model(self):
        """Download or create a weapon detection model."""
        self.logger.info("Setting up weapon detection model...")
        
        # For now, we'll use the standard YOLO model but focus on weapon-like objects
        # In a production system, you'd want a custom-trained weapon detection model
        try:
            # Use YOLOv8s for better accuracy on small objects like weapons
            self.weapon_model = YOLO('yolov8s.pt')
            
            # We'll use the COCO classes that might be weapons or weapon-like
            # Class 43 = scissors, 44 = teddy bear, 45 = hair drier, 46 = toothbrush
            # We'll treat scissors as a potential weapon and add custom logic
            
            self.logger.info("Using YOLOv8s model for weapon-like object detection")
            
            # TODO: In a real implementation, download a custom weapon detection model
            # self._download_custom_weapon_model()
            
        except Exception as e:
            self.logger.error(f"Failed to set up weapon detection: {e}")
            raise
    
    def _download_custom_weapon_model(self):
        """Download a custom weapon detection model (placeholder for future implementation)."""
        # This is where you would download a custom-trained weapon detection model
        # For example, from Roboflow, or a custom trained model
        
        model_urls = {
            # These are example URLs - replace with actual weapon detection models
            "weapon_v1.pt": "https://github.com/ultralytics/yolov5/releases/download/v1.0/weapon.pt",
        }
        
        for model_name, url in model_urls.items():
            model_path = self.model_dir / model_name
            
            if not model_path.exists():
                try:
                    self.logger.info(f"Downloading {model_name}...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    self.logger.info(f"Downloaded {model_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not download {model_name}: {e}")
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.5) -> Dict:
        """
        Detect people and weapons in an image.
        
        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing detection results
        """
        results = {
            'persons': [],
            'weapons': [],
            'crimes': [],  # Person + weapon combinations
            'timestamp': None
        }
        
        try:
            # Detect persons
            person_results = self.person_model(image, conf=conf_threshold, classes=[0])  # Class 0 = person
            
            if person_results[0].boxes is not None:
                for box in person_results[0].boxes:
                    confidence = float(box.conf.cpu().numpy()[0])
                    if confidence >= conf_threshold:
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        
                        person = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'class': 'person'
                        }
                        results['persons'].append(person)
            
            # Detect potential weapons
            weapon_results = self._detect_weapons(image, conf_threshold * 0.8)  # Lower threshold for weapons
            results['weapons'].extend(weapon_results)
            
            # Check for crime scenarios (person + weapon proximity)
            crimes = self._detect_crimes(results['persons'], results['weapons'])
            results['crimes'] = crimes
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
        
        return results
    
    def _detect_weapons(self, image: np.ndarray, conf_threshold: float) -> List[Dict]:
        """Detect weapons or weapon-like objects in the image."""
        weapons = []
        
        try:
            # Use the weapon model to detect objects
            weapon_results = self.weapon_model(image, conf=conf_threshold)
            
            if weapon_results[0].boxes is not None:
                for i, box in enumerate(weapon_results[0].boxes):
                    class_id = int(weapon_results[0].boxes.cls[i].cpu().numpy())
                    class_name = weapon_results[0].names[class_id]
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    # Check if this could be a weapon
                    if self._is_potential_weapon(class_name, class_id):
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        
                        weapon = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'class': class_name,
                            'weapon_type': self._classify_weapon_type(class_name)
                        }
                        weapons.append(weapon)
            
        except Exception as e:
            self.logger.error(f"Error in weapon detection: {e}")
        
        return weapons
    
    def _is_potential_weapon(self, class_name: str, class_id: int) -> bool:
        """Determine if a detected object could be a weapon."""
        # COCO classes that might be weapons
        weapon_class_ids = [43, 37]  # scissors, sports ball (sometimes baseball bats are misclassified)
        weapon_keywords = [
            'knife', 'gun', 'pistol', 'rifle', 'weapon', 'scissors', 'blade',
            'bat', 'baseball', 'stick', 'club', 'rod', 'pole', 'staff'
        ]
        
        # Check class ID
        if class_id in weapon_class_ids:
            return True
        
        # Check class name
        class_name_lower = class_name.lower()
        for keyword in weapon_keywords:
            if keyword in class_name_lower:
                return True
        
        # Additional logic for detecting long cylindrical objects that could be bats/sticks
        if self._looks_like_bat_or_stick(class_name_lower):
            return True
        
        return False
    
    def _looks_like_bat_or_stick(self, class_name_lower: str) -> bool:
        """Check if object name suggests it could be a bat or stick-like weapon."""
        bat_indicators = [
            'baseball', 'bat', 'stick', 'club', 'rod', 'pole', 'staff', 
            'baton', 'cane', 'pipe', 'tube', 'bar'
        ]
        return any(indicator in class_name_lower for indicator in bat_indicators)
    
    def _classify_weapon_type(self, class_name: str) -> str:
        """Classify the type of weapon detected."""
        class_name_lower = class_name.lower()
        
        if any(word in class_name_lower for word in ['gun', 'pistol', 'rifle']):
            return 'firearm'
        elif any(word in class_name_lower for word in ['knife', 'blade', 'scissors']):
            return 'blade'
        elif any(word in class_name_lower for word in ['bat', 'stick', 'club', 'rod', 'pole', 'staff', 'baton']):
            return 'blunt_weapon'
        else:
            return 'unknown'
    
    def _detect_crimes(self, persons: List[Dict], weapons: List[Dict], proximity_threshold: float = 100) -> List[Dict]:
        """
        Detect potential crime scenarios where a person is near a weapon.
        
        Args:
            persons: List of detected persons
            weapons: List of detected weapons
            proximity_threshold: Maximum distance between person and weapon (pixels)
            
        Returns:
            List of crime detection dictionaries
        """
        crimes = []
        
        for person in persons:
            person_center = person['center']
            
            for weapon in weapons:
                weapon_center = weapon['center']
                
                # Calculate distance between person and weapon
                distance = np.sqrt(
                    (person_center[0] - weapon_center[0]) ** 2 +
                    (person_center[1] - weapon_center[1]) ** 2
                )
                
                if distance <= proximity_threshold:
                    crime = {
                        'person': person,
                        'weapon': weapon,
                        'distance': distance,
                        'confidence': min(person['confidence'], weapon['confidence']),
                        'severity': self._calculate_crime_severity(weapon['weapon_type']),
                        'bbox': self._merge_bboxes(person['bbox'], weapon['bbox'])
                    }
                    crimes.append(crime)
        
        return crimes
    
    def _calculate_crime_severity(self, weapon_type: str) -> str:
        """Calculate severity based on weapon type."""
        severity_map = {
            'firearm': 'critical',
            'blade': 'high',
            'blunt_weapon': 'high',  # Baseball bats and clubs are dangerous
            'unknown': 'medium'
        }
        return severity_map.get(weapon_type, 'medium')
    
    def _merge_bboxes(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """Merge two bounding boxes into one that encompasses both."""
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        return [x1, y1, x2, y2]
    
    def draw_detections(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection results on the image.
        
        Args:
            image: Input image
            results: Detection results from detect_objects()
            
        Returns:
            Image with drawn detections
        """
        annotated_image = image.copy()
        
        # Draw persons in green
        for person in results['persons']:
            bbox = person['bbox']
            cv2.rectangle(
                annotated_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),  # Green
                2
            )
            cv2.putText(
                annotated_image,
                f"Person: {person['confidence']:.2f}",
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # Draw weapons in orange
        for weapon in results['weapons']:
            bbox = weapon['bbox']
            cv2.rectangle(
                annotated_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 165, 255),  # Orange
                2
            )
            cv2.putText(
                annotated_image,
                f"{weapon['class']}: {weapon['confidence']:.2f}",
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1
            )
        
        # Draw crimes in red with special annotation
        for crime in results['crimes']:
            bbox = crime['bbox']
            
            # Draw thick red box
            cv2.rectangle(
                annotated_image,
                (int(bbox[0]) - 5, int(bbox[1]) - 5),
                (int(bbox[2]) + 5, int(bbox[3]) + 5),
                (0, 0, 255),  # Red
                5
            )
            
            # Draw "CRIME DETECTED" label
            label = "CRIME DETECTED"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(
                annotated_image,
                (int(bbox[0]) - 5, int(bbox[1]) - 35),
                (int(bbox[0]) + label_size[0] + 10, int(bbox[1]) - 5),
                (0, 0, 255),  # Red background
                -1
            )
            
            # White text
            cv2.putText(
                annotated_image,
                label,
                (int(bbox[0]), int(bbox[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),  # White
                2
            )
        
        return annotated_image


if __name__ == "__main__":
    # Test the weapon detector
    detector = WeaponDetector()
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    results = detector.detect_objects(test_image)
    print(f"Detected {len(results['persons'])} persons, {len(results['weapons'])} weapons, {len(results['crimes'])} crimes")
