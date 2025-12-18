"""
Detection API Resources
"""

import os
import cv2
import numpy as np
from flask import request, current_app
from flask_restful import Resource
from werkzeug.utils import secure_filename
from datetime import datetime
import base64

from services.detection import detection_service


def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


class DetectionResource(Resource):
    """
    POST /api/detect
    Detect objects in a single image
    """
    
    def post(self):
        """
        Detect objects in an uploaded image.
        
        Accepts:
            - multipart/form-data with 'image' file
            - JSON with 'image_base64' (base64 encoded image)
            
        Optional params:
            - confidence: float (default 0.25)
            - iou: float (default 0.45)
            - draw_boxes: bool (default False) - return image with boxes drawn
        """
        try:
            confidence = float(request.form.get('confidence', request.json.get('confidence', 0.25) if request.is_json else 0.25))
            iou = float(request.form.get('iou', request.json.get('iou', 0.45) if request.is_json else 0.45))
            draw_boxes = request.form.get('draw_boxes', 'false').lower() == 'true'
            
            image = None
            
            # Check for file upload
            if 'image' in request.files:
                file = request.files['image']
                if file.filename == '':
                    return {'success': False, 'message': 'No file selected'}, 400
                
                if not allowed_file(file.filename):
                    return {'success': False, 'message': 'File type not allowed'}, 400
                
                # Read image from file
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
            # Check for base64 image
            elif request.is_json and 'image_base64' in request.json:
                image_data = request.json['image_base64']
                # Remove data URL prefix if present
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            else:
                return {'success': False, 'message': 'No image provided'}, 400
            
            if image is None:
                return {'success': False, 'message': 'Could not decode image'}, 400
            
            # Perform detection
            detections = detection_service.detect(
                image,
                confidence_threshold=confidence,
                iou_threshold=iou
            )
            
            response = {
                'success': True,
                'data': {
                    'image_size': {
                        'width': image.shape[1],
                        'height': image.shape[0]
                    },
                    'num_detections': len(detections),
                    'detections': detections
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            # Optionally return image with boxes drawn
            if draw_boxes and detections:
                annotated = detection_service.draw_detections(image, detections)
                _, buffer = cv2.imencode('.jpg', annotated)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                response['data']['annotated_image'] = f"data:image/jpeg;base64,{image_base64}"
            
            return response, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class BatchDetectionResource(Resource):
    """
    POST /api/detect/batch
    Detect objects in multiple images
    """
    
    def post(self):
        """
        Detect objects in multiple uploaded images.
        
        Accepts:
            - multipart/form-data with multiple 'images' files
        """
        try:
            confidence = float(request.form.get('confidence', 0.25))
            iou = float(request.form.get('iou', 0.45))
            
            if 'images' not in request.files:
                return {'success': False, 'message': 'No images provided'}, 400
            
            files = request.files.getlist('images')
            
            if not files:
                return {'success': False, 'message': 'No images provided'}, 400
            
            results = []
            
            for file in files:
                try:
                    if file.filename == '' or not allowed_file(file.filename):
                        results.append({
                            'filename': file.filename,
                            'success': False,
                            'error': 'Invalid file',
                            'detections': []
                        })
                        continue
                    
                    # Read image
                    file_bytes = file.read()
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        results.append({
                            'filename': file.filename,
                            'success': False,
                            'error': 'Could not decode image',
                            'detections': []
                        })
                        continue
                    
                    # Detect
                    detections = detection_service.detect(
                        image,
                        confidence_threshold=confidence,
                        iou_threshold=iou
                    )
                    
                    results.append({
                        'filename': file.filename,
                        'success': True,
                        'image_size': {
                            'width': image.shape[1],
                            'height': image.shape[0]
                        },
                        'num_detections': len(detections),
                        'detections': detections
                    })
                    
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': str(e),
                        'detections': []
                    })
            
            return {
                'success': True,
                'data': {
                    'total_images': len(results),
                    'successful': sum(1 for r in results if r['success']),
                    'results': results
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500
