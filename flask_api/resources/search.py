"""
Similarity Search API Resources
"""

import cv2
import numpy as np
from flask import request, current_app
from flask_restful import Resource
from datetime import datetime
import base64

from services.descriptors import feature_extractor
from services.similarity import similarity_service
from services.detection import detection_service


def decode_image(request_data):
    """Decode image from request (file or base64)"""
    image = None
    
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    elif request.is_json and 'image_base64' in request.json:
        image_data = request.json['image_base64']
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image


class SimilaritySearchResource(Resource):
    """
    POST /api/search/similar
    Search for similar images based on feature vectors
    """
    
    def post(self):
        """
        Find similar images to the uploaded query image.
        
        Accepts:
            - image: Query image (file or base64)
            - top_k: Number of results (default 10)
            - metric: Distance metric (cosine, euclidean, manhattan, chi_square, intersection)
            - filter_class: Optional class name to filter results
        """
        try:
            image = decode_image(request)
            
            if image is None:
                return {'success': False, 'message': 'No valid image provided'}, 400
            
            # Get parameters
            if request.is_json:
                top_k = int(request.json.get('top_k', 10))
                metric = request.json.get('metric', 'cosine')
                filter_class = request.json.get('filter_class', None)
            else:
                top_k = int(request.form.get('top_k', 10))
                metric = request.form.get('metric', 'cosine')
                filter_class = request.form.get('filter_class', None)
            
            # Validate metric
            if metric not in similarity_service.SUPPORTED_METRICS:
                return {
                    'success': False,
                    'message': f'Invalid metric. Supported: {similarity_service.SUPPORTED_METRICS}'
                }, 400
            
            # Extract feature vector
            query_vector = feature_extractor.extract_feature_vector(image)
            
            # Check if index is empty
            if similarity_service.get_index_size() == 0:
                return {
                    'success': True,
                    'data': {
                        'query_vector_length': len(query_vector),
                        'index_size': 0,
                        'results': [],
                        'message': 'Index is empty. Please index some images first.'
                    },
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }, 200
            
            # Search
            results = similarity_service.search(
                query_vector,
                top_k=top_k,
                metric=metric,
                filter_class=filter_class
            )
            
            return {
                'success': True,
                'data': {
                    'query_image_size': {
                        'width': image.shape[1],
                        'height': image.shape[0]
                    },
                    'metric': metric,
                    'top_k': top_k,
                    'index_size': similarity_service.get_index_size(),
                    'num_results': len(results),
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


class ObjectSearchResource(Resource):
    """
    POST /api/search/by-object
    Search for images containing similar objects
    """
    
    def post(self):
        """
        Detect objects in query image, then search for similar objects.
        
        Accepts:
            - image: Query image
            - object_id: Index of detected object to search for (default 0 = first)
            - top_k: Number of results
            - metric: Distance metric
        """
        try:
            image = decode_image(request)
            
            if image is None:
                return {'success': False, 'message': 'No valid image provided'}, 400
            
            # Get parameters
            if request.is_json:
                object_id = int(request.json.get('object_id', 0))
                top_k = int(request.json.get('top_k', 10))
                metric = request.json.get('metric', 'cosine')
                confidence = float(request.json.get('confidence', 0.25))
            else:
                object_id = int(request.form.get('object_id', 0))
                top_k = int(request.form.get('top_k', 10))
                metric = request.form.get('metric', 'cosine')
                confidence = float(request.form.get('confidence', 0.25))
            
            # Detect objects
            detections = detection_service.detect(image, confidence_threshold=confidence)
            
            if not detections:
                return {
                    'success': True,
                    'data': {
                        'num_detections': 0,
                        'message': 'No objects detected in query image',
                        'results': []
                    },
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }, 200
            
            if object_id >= len(detections):
                return {
                    'success': False,
                    'message': f'object_id {object_id} out of range. Found {len(detections)} objects.'
                }, 400
            
            # Get selected object
            selected_detection = detections[object_id]
            bbox = selected_detection['bbox']
            
            # Crop object region
            cropped = image[
                bbox['y_min']:bbox['y_max'],
                bbox['x_min']:bbox['x_max']
            ]
            
            if cropped.size == 0:
                return {'success': False, 'message': 'Empty crop region'}, 400
            
            # Extract features from cropped object
            query_vector = feature_extractor.extract_feature_vector(cropped)
            
            # Search with optional class filter
            results = similarity_service.search(
                query_vector,
                top_k=top_k,
                metric=metric,
                filter_class=selected_detection['class_name']
            )
            
            # Encode cropped object as base64
            _, buffer = cv2.imencode('.jpg', cropped)
            cropped_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': True,
                'data': {
                    'query_image_size': {
                        'width': image.shape[1],
                        'height': image.shape[0]
                    },
                    'num_detections': len(detections),
                    'all_detections': detections,
                    'selected_object': {
                        'id': object_id,
                        'detection': selected_detection,
                        'cropped_image': f"data:image/jpeg;base64,{cropped_base64}"
                    },
                    'metric': metric,
                    'top_k': top_k,
                    'num_results': len(results),
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


class IndexManagementResource(Resource):
    """
    GET/POST/DELETE /api/search/index
    Manage the similarity search index
    """
    
    def get(self):
        """Get index statistics"""
        return {
            'success': True,
            'data': {
                'index_size': similarity_service.get_index_size(),
                'supported_metrics': similarity_service.SUPPORTED_METRICS
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }, 200
    
    def post(self):
        """
        Add image(s) to the index.
        
        Accepts:
            - image: Single image
            - image_id: Unique identifier
            - metadata: Optional metadata (class_name, etc.)
        """
        try:
            image = decode_image(request)
            
            if image is None:
                return {'success': False, 'message': 'No valid image provided'}, 400
            
            if request.is_json:
                image_id = request.json.get('image_id')
                metadata = request.json.get('metadata', {})
            else:
                image_id = request.form.get('image_id')
                metadata = {}
            
            if not image_id:
                return {'success': False, 'message': 'image_id is required'}, 400
            
            # Extract features
            feature_vector = feature_extractor.extract_feature_vector(image)
            
            # Add to index
            similarity_service.add_to_index(image_id, feature_vector, metadata)
            
            return {
                'success': True,
                'data': {
                    'image_id': image_id,
                    'feature_vector_length': len(feature_vector),
                    'index_size': similarity_service.get_index_size()
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500
    
    def delete(self):
        """Clear the index or remove specific image"""
        try:
            if request.is_json:
                image_id = request.json.get('image_id')
            else:
                image_id = request.args.get('image_id')
            
            if image_id:
                similarity_service.remove_from_index(image_id)
                message = f'Removed {image_id} from index'
            else:
                similarity_service.clear_index()
                message = 'Index cleared'
            
            return {
                'success': True,
                'message': message,
                'data': {
                    'index_size': similarity_service.get_index_size()
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500
