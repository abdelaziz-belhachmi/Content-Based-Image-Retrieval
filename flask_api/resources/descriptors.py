"""
Descriptor Extraction API Resources
"""

import cv2
import numpy as np
from flask import request, current_app
from flask_restful import Resource
from datetime import datetime
import base64

from services.descriptors import feature_extractor


def decode_image(request_data):
    """Decode image from request (file or base64)"""
    image = None
    
    # Check for file upload
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check for base64
    elif request.is_json and 'image_base64' in request.json:
        image_data = request.json['image_base64']
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image


class DescriptorExtractionResource(Resource):
    """
    POST /api/descriptors/extract
    Extract all visual descriptors from an image
    """
    
    def post(self):
        """
        Extract visual descriptors from an uploaded image.
        
        Accepts:
            - multipart/form-data with 'image' file
            - JSON with 'image_base64'
            
        Optional params:
            - features: comma-separated list of features to extract
                       (color, texture, shape, all)
        """
        try:
            image = decode_image(request)
            
            if image is None:
                return {'success': False, 'message': 'No valid image provided'}, 400
            
            # Get requested features
            features_param = request.form.get(
                'features', 
                request.json.get('features', 'all') if request.is_json else 'all'
            )
            
            if features_param == 'all':
                # Extract all features
                features = feature_extractor.extract_all_features(image)
            else:
                requested = [f.strip().lower() for f in features_param.split(',')]
                features = {}
                
                if 'color' in requested:
                    features['color_histogram_hsv'] = feature_extractor.extract_color_histogram_hsv(image).tolist()
                    features['dominant_colors'] = feature_extractor.extract_dominant_colors(image)
                    features['color_moments'] = feature_extractor.extract_color_moments(image).tolist()
                
                if 'texture' in requested:
                    features['tamura'] = feature_extractor.extract_tamura_features(image)
                    features['gabor'] = feature_extractor.extract_gabor_features(image).tolist()
                    features['lbp'] = feature_extractor.extract_lbp_features(image).tolist()
                    features['glcm'] = feature_extractor.extract_glcm_features(image)
                
                if 'shape' in requested:
                    features['hu_moments'] = feature_extractor.extract_hu_moments(image).tolist()
                    features['contour'] = feature_extractor.extract_contour_features(image)
                    features['hog'] = feature_extractor.extract_hog_features(image).tolist()
            
            return {
                'success': True,
                'data': {
                    'image_size': {
                        'width': image.shape[1],
                        'height': image.shape[0]
                    },
                    'features': features
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class ObjectDescriptorResource(Resource):
    """
    POST /api/descriptors/extract/object
    Extract descriptors from a specific region (cropped object)
    """
    
    def post(self):
        """
        Extract descriptors from a cropped region of an image.
        
        Accepts:
            - image: The full image
            - bbox: Bounding box {x_min, y_min, x_max, y_max}
        """
        try:
            image = decode_image(request)
            
            if image is None:
                return {'success': False, 'message': 'No valid image provided'}, 400
            
            # Get bounding box
            if request.is_json:
                bbox = request.json.get('bbox', {})
            else:
                bbox = {
                    'x_min': int(request.form.get('x_min', 0)),
                    'y_min': int(request.form.get('y_min', 0)),
                    'x_max': int(request.form.get('x_max', image.shape[1])),
                    'y_max': int(request.form.get('y_max', image.shape[0]))
                }
            
            # Validate bbox
            x_min = max(0, bbox.get('x_min', 0))
            y_min = max(0, bbox.get('y_min', 0))
            x_max = min(image.shape[1], bbox.get('x_max', image.shape[1]))
            y_max = min(image.shape[0], bbox.get('y_max', image.shape[0]))
            
            if x_max <= x_min or y_max <= y_min:
                return {'success': False, 'message': 'Invalid bounding box'}, 400
            
            # Crop image
            cropped = image[y_min:y_max, x_min:x_max]
            
            if cropped.size == 0:
                return {'success': False, 'message': 'Empty crop region'}, 400
            
            # Extract features
            features = feature_extractor.extract_all_features(cropped)
            
            # Also get feature vector for similarity
            feature_vector = feature_extractor.extract_feature_vector(cropped).tolist()
            
            return {
                'success': True,
                'data': {
                    'bbox': {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max
                    },
                    'crop_size': {
                        'width': x_max - x_min,
                        'height': y_max - y_min
                    },
                    'features': features,
                    'feature_vector': feature_vector,
                    'feature_vector_length': len(feature_vector)
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class StoredDescriptorResource(Resource):
    """
    GET /api/descriptors/<image_id>
    Retrieve stored descriptors for an image
    """
    
    def get(self, image_id):
        """
        Get stored descriptors for an image.
        
        This is a placeholder - actual implementation would query database.
        """
        try:
            from services.similarity import similarity_service
            
            if image_id not in similarity_service.feature_index:
                return {
                    'success': False,
                    'message': f'No descriptors found for image {image_id}'
                }, 404
            
            return {
                'success': True,
                'data': {
                    'image_id': image_id,
                    'feature_vector': similarity_service.feature_index[image_id].tolist(),
                    'metadata': similarity_service.metadata.get(image_id, {})
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500
