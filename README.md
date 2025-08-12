# Real-Time Threat Detection System

A specialized deep learning system designed for **real-time crime detection** using webcam feeds. The system detects people and weapons, and generates **"CRIME DETECTED"** alerts when weapons are found near people.

## Project Information

**Created By:** Fahd Shah, 2025  
**Institution:** Loughborough University  

**Academic Context:**  
Designed as apart of a MSc Systems Engineering Thesis Project:  
*"Predictive Policing: Balancing Efficiency & Ethics in AI-Enhanced Crime Prevention"*

## Features

- ** Person Detection**: Accurate identification of people in real-time
- ** Weapon Detection**: Detects guns, knives, scissors, and other weapons
- ** Crime Detection**: Special "CRIME DETECTED" alerts when person + weapon detected together
- ** Real-Time Processing**: Live webcam feed processing at 30 FPS
- ** Audio Alerts**: Sound notifications for threat detection
- ** Live Statistics**: Real-time detection counters and performance metrics

## Project Structure

```
threat_detection_system/
├── src/
│   ├── models/           # Deep learning models
│   ├── detection/        # Detection logic
│   ├── utils/           # Utility functions
│   └── config/          # Configuration files
├── data/
│   ├── models/          # Pre-trained model weights
│   └── samples/         # Sample images/videos
├── screenshots/          # Screenshots from detection sessions
├── tests/               # Unit tests
└── requirements.txt     # Python dependencies
```

## Quick Start

1. **Install Dependencies:**
   ```bash
   py -3 quick_start.py
   ```

2. **Start Threat Detection:**
   ```bash
   py -3 start_threat_detection.py
   ```
   
3. **Controls:**
   - `Q` or `ESC` - Quit the system
   - `S` - Save screenshot
   - `R` - Reset statistics

## Screenshots

The system automatically saves screenshots when:
- **Manual**: Press `S` key during detection to capture current frame
- **Automatic**: Screenshots are saved when threats are detected (crime alerts)

**Screenshot Location:**
- All screenshots are saved in the `screenshots/` folder
- Naming format: `threat_detection_screenshot_YYYYMMDD_HHMMSS.jpg`
- Include detection boxes, confidence scores, and status overlay
- Serve as evidence with timestamp information
## Configuration

Edit `src/config/config.yaml` to customize:
- Detection thresholds
- Target object classes
- Alert settings
- Input/output preferences

## Usage

### Basic Detection
```python
from src.detection.detector import ThreatDetector

detector = ThreatDetector()
results = detector.detect_image("path/to/image.jpg")
```

### Real-time Video Processing
```python
detector.process_video_stream(camera_id=0)
```

## Model Information

- Base Model: YOLOv8 (ultralytics)
- Person Detection: COCO-trained weights
- Threat Objects: Custom fine-tuned model
- Inference Speed: ~30-60 FPS (depending on hardware)

## Security Considerations

This tool is designed for legitimate security applications and research purposes. Ensure compliance with:
- Local privacy laws
- Surveillance regulations
- Ethical AI guidelines
- Data protection requirements

## Academic Disclaimer

This system was developed as part of academic research into predictive policing technologies. The focus of the associated thesis work examines both the efficiency benefits and ethical implications of AI-enhanced crime prevention systems. Users should consider the broader societal impacts when implementing such technologies.

---

*"The pursuit of knowledge and innovation should always be balanced with ethical responsibility and human dignity."*  
*Ad Astra* ⭐
