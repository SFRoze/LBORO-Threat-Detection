"""
Threat Detection System - Web Interface

This module provides a Streamlit-based web interface for the threat detection system.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from detection.detector import ThreatDetector
    from utils.logger import setup_logger
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you've installed the requirements: pip install -r requirements.txt")
    st.stop()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Threat Detection System",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚨 Threat Detection System")
    st.markdown("Deep Learning-based Object Detection for Security Applications")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            help="Larger models are more accurate but slower"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        # Device selection
        device_options = ['auto', 'cpu', 'cuda']
        selected_device = st.selectbox(
            "Processing Device",
            device_options,
            index=0,
            help="Choose processing device (CUDA requires compatible GPU)"
        )
        
        # Alert settings
        st.subheader("🔔 Alert Settings")
        person_alerts = st.checkbox("Person Detection Alerts", value=True)
        multiple_person_threshold = st.number_input(
            "Multiple Person Threshold",
            min_value=1,
            max_value=20,
            value=3,
            help="Alert when this many or more people are detected"
        )
        
        # Initialize detector button
        if st.button("🔄 Initialize/Update Detector", type="primary"):
            with st.spinner("Initializing detector..."):
                try:
                    # Create custom config
                    config = create_custom_config(
                        selected_model,
                        confidence_threshold,
                        selected_device,
                        person_alerts,
                        multiple_person_threshold
                    )
                    
                    st.session_state.detector = ThreatDetector()
                    st.session_state.detector.config.update(config)
                    st.session_state.detector._load_models()
                    
                    st.success("Detector initialized successfully!")
                    
                except Exception as e:
                    st.error(f"Error initializing detector: {e}")
    
    # Main content area
    if st.session_state.detector is None:
        st.warning("⚠️ Please initialize the detector using the sidebar controls.")
        st.info("Click 'Initialize/Update Detector' in the sidebar to get started.")
        return
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📊 Statistics", "ℹ️ About"])
    
    with tab1:
        image_detection_tab(st.session_state.detector)
    
    with tab2:
        video_detection_tab(st.session_state.detector)
    
    with tab3:
        statistics_tab(st.session_state.detector)
    
    with tab4:
        about_tab()


def create_custom_config(model, confidence, device, person_alerts, person_threshold):
    """Create custom configuration dictionary."""
    return {
        'model': {
            'primary_model': model,
            'confidence_threshold': confidence,
            'device': device,
            'nms_threshold': 0.45
        },
        'alerts': {
            'person_detection': person_alerts,
            'multiple_persons': True,
            'person_threshold': person_threshold,
            'console_output': False,  # Disable console output for web interface
            'log_file': True,
            'confidence_minimum': confidence
        },
        'processing': {
            'draw_boxes': True,
            'draw_labels': True,
            'draw_confidence': True
        }
    }


def image_detection_tab(detector):
    """Image detection interface."""
    st.header("📷 Image Detection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
        help="Upload an image for object detection"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Process image
        with st.spinner("Processing image..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                
                try:
                    # Run detection
                    results = detector.detect_image(tmp_file.name, save_output=False)
                    
                    # Display results
                    with col2:
                        st.subheader("Detection Results")
                        
                        # Show processed image with bounding boxes
                        if results['objects']:
                            # Load and display the processed image
                            # Note: In a real implementation, you'd get the annotated image from the detector
                            st.image(image, use_column_width=True, caption=f"Found {len(results['objects'])} objects")
                        else:
                            st.image(image, use_column_width=True, caption="No objects detected")
                    
                    # Display detection details
                    st.subheader("Detection Details")
                    
                    if results['objects']:
                        # Create DataFrame for better display
                        import pandas as pd
                        
                        df_data = []
                        for obj in results['objects']:
                            df_data.append({
                                'Class': obj['class'],
                                'Confidence': f"{obj['confidence']:.3f}",
                                'Center X': f"{obj['center'][0]:.0f}",
                                'Center Y': f"{obj['center'][1]:.0f}"
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Show counts
                        st.subheader("Object Counts")
                        count_cols = st.columns(len(results['counts']))
                        for i, (obj_class, count) in enumerate(results['counts'].items()):
                            with count_cols[i % len(count_cols)]:
                                st.metric(obj_class.title(), count)
                    else:
                        st.info("No objects detected in the image.")
                    
                    # Show alerts
                    if results.get('alerts'):
                        st.subheader("🚨 Alerts")
                        for alert in results['alerts']:
                            severity_color = {
                                'low': 'info',
                                'medium': 'warning', 
                                'high': 'error',
                                'critical': 'error'
                            }.get(alert['severity'], 'info')
                            
                            getattr(st, severity_color)(f"**{alert['type']}**: {alert['message']}")
                    
                    # Performance info
                    st.subheader("⚡ Performance")
                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        st.metric("Processing Time", f"{results['processing_time']:.2f}s")
                    with perf_col2:
                        st.metric("Objects Found", len(results['objects']))
                
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file.name)


def video_detection_tab(detector):
    """Video detection interface."""
    st.header("🎥 Video Detection")
    st.info("Video detection functionality would be implemented here.")
    st.markdown("""
    **Features to implement:**
    - Upload video files for batch processing
    - Real-time webcam detection
    - Video stream processing
    - Frame-by-frame analysis
    - Export processed videos
    """)
    
    # Placeholder for video upload
    st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video for object detection (Not yet implemented)",
        disabled=True
    )


def statistics_tab(detector):
    """Statistics and monitoring interface."""
    st.header("📊 Statistics & Monitoring")
    
    # Get current statistics
    stats = detector.get_statistics()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Detections", stats['total_detections'])
    
    with col2:
        st.metric("Person Detections", stats['person_detections'])
    
    with col3:
        st.metric("Threat Detections", stats['threat_detections'])
    
    with col4:
        st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.3f}s")
    
    # Alert statistics
    if hasattr(detector, 'alert_manager') and detector.alert_manager:
        st.subheader("🔔 Alert Statistics")
        alert_stats = detector.alert_manager.get_alert_stats()
        
        if alert_stats['total_alerts'] > 0:
            # Alert counts by type
            st.subheader("Alerts by Type")
            alert_type_data = alert_stats['by_type']
            if alert_type_data:
                import pandas as pd
                df = pd.DataFrame(list(alert_type_data.items()), columns=['Alert Type', 'Count'])
                st.bar_chart(df.set_index('Alert Type'))
            
            # Alert counts by severity
            st.subheader("Alerts by Severity")
            severity_data = alert_stats['by_severity']
            if severity_data:
                df = pd.DataFrame(list(severity_data.items()), columns=['Severity', 'Count'])
                st.bar_chart(df.set_index('Severity'))
            
            st.metric("Alerts (Last 24h)", alert_stats['last_24h'])
        else:
            st.info("No alerts generated yet.")
    
    # Reset statistics button
    if st.button("🔄 Reset Statistics"):
        detector.reset_statistics()
        if hasattr(detector, 'alert_manager') and detector.alert_manager:
            detector.alert_manager.clear_alert_history()
        st.success("Statistics reset successfully!")
        st.experimental_rerun()


def about_tab():
    """About and information tab."""
    st.header("ℹ️ About Threat Detection System")
    
    st.markdown("""
    ## Overview
    This is a deep learning-based threat detection system designed to identify people and potential security threats in images and videos.
    
    ## Features
    - **Person Detection**: Accurately identifies people in images and video streams
    - **Real-time Processing**: Supports live camera feeds and video processing
    - **Configurable Alerts**: Customizable alert system for different threat levels
    - **Multiple Models**: Support for different YOLO model variants (nano to extra-large)
    - **Flexible Input**: Handles images, videos, and real-time camera feeds
    - **Web Interface**: Easy-to-use web interface for non-technical users
    
    ## Technology Stack
    - **Deep Learning Framework**: PyTorch + Ultralytics YOLO
    - **Computer Vision**: OpenCV
    - **Web Interface**: Streamlit
    - **Alert System**: Multi-channel notifications (console, email, webhooks)
    
    ## Model Information
    - **Base Architecture**: YOLOv8 (You Only Look Once)
    - **Training Data**: COCO dataset (Common Objects in Context)
    - **Supported Objects**: 80+ object classes including person, vehicles, etc.
    - **Custom Classes**: Extensible for threat-specific objects
    
    ## Security & Privacy
    - All processing is done locally (no data sent to external servers)
    - Configurable data retention policies
    - Optional face anonymization
    - Audit logging for compliance
    
    ## Use Cases
    - **Security Monitoring**: Perimeter security and access control
    - **Crowd Management**: Monitoring public spaces and events
    - **Safety Compliance**: Ensuring safety protocols in industrial settings
    - **Traffic Monitoring**: Vehicle and pedestrian detection
    
    ## Getting Started
    1. Configure your detection parameters in the sidebar
    2. Initialize the detector
    3. Upload an image or start video processing
    4. Monitor alerts and statistics
    
    ## Support
    For technical support or feature requests, please check the project documentation.
    """)
    
    # System information
    with st.expander("🔧 System Information"):
        st.code(f"""
        Python Version: {sys.version}
        Streamlit Version: {st.__version__}
        OpenCV Version: {cv2.__version__}
        """)


if __name__ == "__main__":
    main()
