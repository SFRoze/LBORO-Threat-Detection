#!/usr/bin/env python3
"""
Basic Tests for Threat Detection System

This module contains basic unit tests for the threat detection system.
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from detection.detector import ThreatDetector
    from utils.alerts import AlertManager
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you've installed the requirements and the src directory is properly structured")
    sys.exit(1)


class TestThreatDetector(unittest.TestCase):
    """Test cases for the ThreatDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        print("Setting up ThreatDetector tests...")
        cls.detector = None
    
    def setUp(self):
        """Set up before each test method."""
        if self.detector is None:
            try:
                self.__class__.detector = ThreatDetector()
                print("✅ ThreatDetector initialized successfully")
            except Exception as e:
                self.skipTest(f"Could not initialize ThreatDetector: {e}")
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
        self.assertIsInstance(self.detector.config, dict)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = self.detector.config
        
        # Check required config sections
        self.assertIn('model', config)
        self.assertIn('classes', config)
        self.assertIn('alerts', config)
        self.assertIn('processing', config)
        
        # Check model config
        model_config = config['model']
        self.assertIn('primary_model', model_config)
        self.assertIn('confidence_threshold', model_config)
        self.assertIn('device', model_config)
    
    def test_statistics_functions(self):
        """Test statistics tracking functions."""
        # Get initial stats
        initial_stats = self.detector.get_statistics()
        self.assertIsInstance(initial_stats, dict)
        self.assertIn('total_detections', initial_stats)
        self.assertIn('person_detections', initial_stats)
        self.assertIn('processing_time', initial_stats)
        
        # Reset statistics
        self.detector.reset_statistics()
        reset_stats = self.detector.get_statistics()
        self.assertEqual(reset_stats['total_detections'], 0)
    
    def test_detection_with_sample_image(self):
        """Test detection on a synthetic image."""
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:400, 200:400] = [255, 255, 255]  # White rectangle
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_image)
            
            try:
                # Run detection
                results = self.detector.detect_image(tmp_file.name, save_output=False)
                
                # Check results structure
                self.assertIsInstance(results, dict)
                self.assertIn('objects', results)
                self.assertIn('counts', results)
                self.assertIn('timestamp', results)
                self.assertIn('processing_time', results)
                
                # Check that processing time is reasonable (< 30 seconds)
                self.assertLess(results['processing_time'], 30.0)
                
                print(f"✅ Detection completed in {results['processing_time']:.2f}s")
                print(f"Found {len(results['objects'])} objects")
                
            finally:
                # Clean up
                os.unlink(tmp_file.name)


class TestAlertManager(unittest.TestCase):
    """Test cases for the AlertManager class."""
    
    def setUp(self):
        """Set up before each test method."""
        self.alert_config = {
            'person_detection': True,
            'console_output': False,  # Disable console output for tests
            'log_file': False,
            'confidence_minimum': 0.7
        }
        self.alert_manager = AlertManager(self.alert_config)
    
    def test_alert_manager_initialization(self):
        """Test that alert manager initializes correctly."""
        self.assertIsNotNone(self.alert_manager)
        self.assertEqual(self.alert_manager.config, self.alert_config)
        self.assertIsInstance(self.alert_manager.alert_history, list)
    
    def test_person_detection_alert(self):
        """Test person detection alert generation."""
        # Mock detection data with a person
        detections = {
            'source': 'test_image.jpg',
            'objects': [
                {
                    'class': 'person',
                    'confidence': 0.85,
                    'center': [320, 240],
                    'bbox': [200, 100, 440, 380]
                }
            ],
            'counts': {'person': 1}
        }
        
        # Process detections
        alerts = self.alert_manager.process_detections(detections)
        
        # Check that alert was generated
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0]['type'], 'person_detection')
        self.assertIn('person', alerts[0]['message'].lower())
    
    def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        # Test that cooldown prevents duplicate alerts
        result1 = self.alert_manager._check_alert_cooldown('test_alert')
        result2 = self.alert_manager._check_alert_cooldown('test_alert')
        
        # First call should return True, second should return False (in cooldown)
        self.assertTrue(result1)
        self.assertFalse(result2)
    
    def test_alert_statistics(self):
        """Test alert statistics functions."""
        # Get initial stats
        initial_stats = self.alert_manager.get_alert_stats()
        self.assertIsInstance(initial_stats, dict)
        self.assertIn('total_alerts', initial_stats)
        self.assertIn('by_type', initial_stats)
        
        # Clear history
        self.alert_manager.clear_alert_history()
        cleared_stats = self.alert_manager.get_alert_stats()
        self.assertEqual(cleared_stats['total_alerts'], 0)


class TestConfigurationHandling(unittest.TestCase):
    """Test cases for configuration handling."""
    
    def test_default_config_creation(self):
        """Test creation of default configuration when file is missing."""
        detector = ThreatDetector()
        config = detector._get_default_config()
        
        # Check that default config has required sections
        self.assertIsInstance(config, dict)
        self.assertIn('model', config)
        self.assertIn('alerts', config)
        self.assertIn('processing', config)


def run_performance_test():
    """Run a simple performance test."""
    print("\n🚀 Running Performance Test...")
    
    try:
        detector = ThreatDetector()
        
        # Create test images of different sizes
        test_sizes = [(320, 240), (640, 480), (1280, 720)]
        
        for width, height in test_sizes:
            # Create test image
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, test_image)
                
                try:
                    # Time the detection
                    import time
                    start_time = time.time()
                    results = detector.detect_image(tmp_file.name, save_output=False)
                    processing_time = time.time() - start_time
                    
                    print(f"  {width}x{height}: {processing_time:.3f}s ({len(results['objects'])} objects)")
                    
                finally:
                    os.unlink(tmp_file.name)
        
        print("✅ Performance test completed")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


if __name__ == '__main__':
    print("🧪 Threat Detection System - Basic Tests")
    print("=" * 50)
    
    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    run_performance_test()
    
    print("\n✅ All tests completed!")
