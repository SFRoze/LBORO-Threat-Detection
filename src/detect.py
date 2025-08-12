#!/usr/bin/env python3
"""
Threat Detection System - Command Line Interface

This script provides a command-line interface for the threat detection system.
"""

import argparse
import sys
import os
from pathlib import Path
import time

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from detection.detector import ThreatDetector
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you've installed the requirements: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Threat Detection System - Deep Learning Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect objects in an image
  python src/detect.py --image path/to/image.jpg

  # Process a video file
  python src/detect.py --video path/to/video.mp4 --output output/result.mp4

  # Real-time webcam detection
  python src/detect.py --camera 0

  # Use custom configuration
  python src/detect.py --image test.jpg --config custom_config.yaml

  # Batch process multiple images
  python src/detect.py --batch path/to/images/ --output output/

  # Show statistics
  python src/detect.py --image test.jpg --stats
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str,
                           help='Path to input image file')
    input_group.add_argument('--video', '-v', type=str,
                           help='Path to input video file')
    input_group.add_argument('--camera', '-c', type=int,
                           help='Camera index for real-time detection (usually 0)')
    input_group.add_argument('--batch', '-b', type=str,
                           help='Directory containing images for batch processing')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='Output file/directory path')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save output files')
    
    # Model options
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--model', type=str,
                       help='Model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--confidence', type=float,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       help='Processing device')
    
    # Display options
    parser.add_argument('--no-display', action='store_true',
                       help='Don\'t display results (useful for headless systems)')
    parser.add_argument('--stats', action='store_true',
                       help='Show detection statistics')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--verbose', action='store_true',
                       help='Increase output verbosity')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.camera is not None and args.output:
        parser.error("--output is not supported with --camera (real-time mode)")
    
    if args.batch and not args.output:
        parser.error("--output is required when using --batch")
    
    try:
        # Initialize detector
        print("Initializing Threat Detection System...")
        detector = ThreatDetector(config_path=args.config)
        
        # Override config with command line arguments
        if args.model:
            detector.config['model']['primary_model'] = args.model
            detector._load_models()  # Reload with new model
        
        if args.confidence is not None:
            detector.config['model']['confidence_threshold'] = args.confidence
        
        if args.device:
            detector.config['model']['device'] = args.device
            detector._load_models()  # Reload with new device
        
        if args.quiet:
            detector.config['logging']['level'] = 'WARNING'
        elif args.verbose:
            detector.config['logging']['level'] = 'DEBUG'
        
        # Process input based on mode
        if args.image:
            process_image(detector, args)
        elif args.video:
            process_video(detector, args)
        elif args.camera is not None:
            process_camera(detector, args)
        elif args.batch:
            process_batch(detector, args)
        
        # Show statistics if requested
        if args.stats:
            show_statistics(detector)
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def process_image(detector, args):
    """Process a single image."""
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    print(f"Processing image: {args.image}")
    start_time = time.time()
    
    try:
        results = detector.detect_image(
            args.image,
            save_output=not args.no_save
        )
        
        processing_time = time.time() - start_time
        
        # Display results
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Objects detected: {len(results['objects'])}")
        
        if results['objects']:
            print("\nDetections:")
            for obj in results['objects']:
                print(f"  - {obj['class']}: {obj['confidence']:.3f}")
                print(f"    Location: ({obj['center'][0]:.0f}, {obj['center'][1]:.0f})")
        
        # Show alert information
        if results.get('alerts'):
            print(f"\nAlerts generated: {len(results['alerts'])}")
            for alert in results['alerts']:
                severity_symbol = {"low": "ℹ️", "medium": "⚠️", "high": "🚨", "critical": "🔥"}.get(alert['severity'], "❓")
                print(f"  {severity_symbol} {alert['message']}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


def process_video(detector, args):
    """Process a video file."""
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print(f"Processing video: {args.video}")
    
    # Determine output path
    output_path = None
    if args.output and not args.no_save:
        output_path = args.output
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    start_time = time.time()
    
    try:
        results = detector.detect_video(args.video, output_path)
        
        processing_time = time.time() - start_time
        
        # Display results
        print(f"\nVideo processing completed in {processing_time:.2f} seconds")
        print(f"Frames processed: {results['processed_frames']}/{results['total_frames']}")
        print(f"Total persons detected: {results['summary']['total_persons']}")
        print(f"Max persons per frame: {results['summary']['max_persons_per_frame']}")
        
        if output_path:
            print(f"Output saved to: {output_path}")
            
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)


def process_camera(detector, args):
    """Process real-time camera feed."""
    print(f"Starting real-time detection from camera {args.camera}")
    print("Press 'q' to quit")
    
    try:
        detector.process_video_stream(source=args.camera)
        
    except Exception as e:
        print(f"Error processing camera feed: {e}")
        sys.exit(1)


def process_batch(detector, args):
    """Process multiple images in a directory."""
    batch_dir = Path(args.batch)
    if not batch_dir.exists():
        print(f"Error: Batch directory not found: {args.batch}")
        sys.exit(1)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(batch_dir.glob(f"*{ext}"))
        image_files.extend(batch_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in directory: {args.batch}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    total_detections = 0
    start_time = time.time()
    
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            results = detector.detect_image(
                str(image_file),
                save_output=not args.no_save
            )
            
            total_detections += len(results['objects'])
            
            # Save results summary
            if not args.no_save:
                results_file = output_dir / f"{image_file.stem}_results.txt"
                with open(results_file, 'w') as f:
                    f.write(f"Detection results for: {image_file.name}\\n")
                    f.write(f"Processing time: {results['processing_time']:.3f}s\\n")
                    f.write(f"Objects detected: {len(results['objects'])}\\n\\n")
                    
                    for obj in results['objects']:
                        f.write(f"{obj['class']}: {obj['confidence']:.3f}\\n")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    print(f"\\nBatch processing completed in {processing_time:.2f} seconds")
    print(f"Average time per image: {processing_time/len(image_files):.2f} seconds")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(image_files):.1f}")


def show_statistics(detector):
    """Display detection statistics."""
    stats = detector.get_statistics()
    
    print("\\n" + "="*50)
    print("DETECTION STATISTICS")
    print("="*50)
    print(f"Total detections processed: {stats['total_detections']}")
    print(f"Person detections: {stats['person_detections']}")
    print(f"Threat detections: {stats['threat_detections']}")
    print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    print(f"Total processing time: {stats['processing_time']:.2f}s")


if __name__ == "__main__":
    main()
