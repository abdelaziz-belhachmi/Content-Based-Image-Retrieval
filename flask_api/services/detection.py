"""
YOLO Object Detection Service
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class DetectionService:
    """
    Service for object detection using YOLOv8.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to reuse model instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if DetectionService._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        from config import get_config
        config = get_config()
        
        model_path = config.YOLO_MODEL_PATH
        
        # If custom model doesn't exist, use pretrained YOLOv8n
        if not Path(model_path).exists():
            print(f"Custom model not found at {model_path}")
            print("Using pretrained YOLOv8n model")
            model_path = 'yolov8n.pt'
        
        DetectionService._model = YOLO(model_path)
        print(f"Loaded YOLO model: {model_path}")
    
    @property
    def model(self):
        return DetectionService._model
    
    def detect(self, image, confidence_threshold=0.25, iou_threshold=0.45):
        """
        Detect objects in an image.
        
        Args:
            image: numpy array (BGR) or path to image
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detected objects with class, confidence, and bbox
        """
        # Run inference
        results = self.model(
            image,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                detection = {
                    'id': i,
                    'class_id': class_id,
                    'class_name': result.names[class_id],
                    'confidence': round(confidence, 4),
                    'bbox': {
                        'x_min': int(xyxy[0]),
                        'y_min': int(xyxy[1]),
                        'x_max': int(xyxy[2]),
                        'y_max': int(xyxy[3])
                    }
                }
                detections.append(detection)
        
        return detections
    
    def detect_from_file(self, file_path, **kwargs):
        """
        Detect objects from an image file.
        """
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
        return self.detect(image, **kwargs)
    
    def detect_from_bytes(self, image_bytes, **kwargs):
        """
        Detect objects from image bytes.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes")
        return self.detect(image, **kwargs)
    
    def detect_and_crop(self, image, confidence_threshold=0.25, iou_threshold=0.45):
        """
        Detect objects and return cropped regions.
        
        Returns:
            List of tuples (detection_info, cropped_image)
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        detections = self.detect(image, confidence_threshold, iou_threshold)
        
        results = []
        for det in detections:
            bbox = det['bbox']
            crop = image[
                bbox['y_min']:bbox['y_max'],
                bbox['x_min']:bbox['x_max']
            ]
            results.append((det, crop))
        
        return results
    
    def detect_batch(self, images, **kwargs):
        """
        Detect objects in multiple images.
        
        Args:
            images: List of image paths or numpy arrays
            
        Returns:
            List of detection results for each image
        """
        batch_results = []
        
        for image in images:
            try:
                if isinstance(image, (str, Path)):
                    detections = self.detect_from_file(image, **kwargs)
                else:
                    detections = self.detect(image, **kwargs)
                batch_results.append({
                    'success': True,
                    'detections': detections
                })
            except Exception as e:
                batch_results.append({
                    'success': False,
                    'error': str(e),
                    'detections': []
                })
        
        return batch_results
    
    def draw_detections(self, image, detections, line_thickness=2):
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: numpy array (BGR)
            detections: List of detection dictionaries
            line_thickness: Thickness of bounding box lines
            
        Returns:
            Image with drawn detections
        """
        output = image.copy()
        
        # Color palette for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (64, 0, 0), (0, 64, 0), (0, 0, 64)
        ]
        
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            color = colors[class_id % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(
                output,
                (bbox['x_min'], bbox['y_min']),
                (bbox['x_max'], bbox['y_max']),
                color,
                line_thickness
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            cv2.rectangle(
                output,
                (bbox['x_min'], bbox['y_min'] - text_height - 10),
                (bbox['x_min'] + text_width, bbox['y_min']),
                color,
                -1
            )
            
            cv2.putText(
                output,
                label,
                (bbox['x_min'], bbox['y_min'] - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return output


# Singleton instance
detection_service = DetectionService()
