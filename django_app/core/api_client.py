"""
Flask API Client for Django
Handles communication with the Flask REST API
"""

import requests
import base64
from django.conf import settings
from pathlib import Path
import json


class FlaskAPIClient:
    """
    Client for communicating with Flask REST API
    """
    
    def __init__(self, base_url=None):
        self.base_url = base_url or getattr(settings, 'FLASK_API_URL', 'http://localhost:5000/api')
        self.timeout = 120  # seconds (increased for 3D model processing)
    
    def _make_request(self, method, endpoint, **kwargs):
        """Make HTTP request to Flask API"""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Could not connect to Flask API. Make sure it is running.'
            }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request to Flask API timed out.'
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # Detection endpoints
    
    def detect_objects(self, image_path=None, image_bytes=None, confidence=0.25, iou=0.45, draw_boxes=False):
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to image file
            image_bytes: Image bytes (alternative to path)
            confidence: Confidence threshold
            iou: IoU threshold
            draw_boxes: Return annotated image
            
        Returns:
            API response with detections
        """
        if image_path:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'confidence': confidence,
                    'iou': iou,
                    'draw_boxes': str(draw_boxes).lower()
                }
                return self._make_request('POST', '/detect', files=files, data=data)
        elif image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            json_data = {
                'image_base64': image_base64,
                'confidence': confidence,
                'iou': iou
            }
            return self._make_request('POST', '/detect', json=json_data)
        else:
            return {'success': False, 'error': 'No image provided'}
    
    def detect_batch(self, image_paths, confidence=0.25, iou=0.45):
        """Detect objects in multiple images"""
        files = [('images', open(p, 'rb')) for p in image_paths]
        data = {'confidence': confidence, 'iou': iou}
        
        try:
            result = self._make_request('POST', '/detect/batch', files=files, data=data)
        finally:
            for _, f in files:
                f.close()
        
        return result
    
    # Descriptor endpoints
    
    def extract_descriptors(self, image_path=None, image_bytes=None, features='all'):
        """
        Extract visual descriptors from an image.
        
        Args:
            image_path: Path to image file
            image_bytes: Image bytes
            features: Which features to extract ('all', 'color', 'texture', 'shape')
            
        Returns:
            API response with extracted features
        """
        if image_path:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'features': features}
                return self._make_request('POST', '/descriptors/extract', files=files, data=data)
        elif image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            json_data = {
                'image_base64': image_base64,
                'features': features
            }
            return self._make_request('POST', '/descriptors/extract', json=json_data)
        else:
            return {'success': False, 'error': 'No image provided'}
    
    def extract_object_descriptors(self, image_path, bbox):
        """
        Extract descriptors from a cropped region.
        
        Args:
            image_path: Path to image file
            bbox: Dict with x_min, y_min, x_max, y_max
            
        Returns:
            API response with features
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'x_min': bbox['x_min'],
                'y_min': bbox['y_min'],
                'x_max': bbox['x_max'],
                'y_max': bbox['y_max']
            }
            return self._make_request('POST', '/descriptors/extract/object', files=files, data=data)
    
    # Search endpoints
    
    def search_similar(self, image_path=None, image_bytes=None, top_k=10, metric='cosine', filter_class=None):
        """
        Search for similar images.
        
        Args:
            image_path: Query image path
            image_bytes: Query image bytes
            top_k: Number of results
            metric: Distance metric
            filter_class: Optional class filter
            
        Returns:
            API response with similar images
        """
        if image_path:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'top_k': top_k,
                    'metric': metric,
                }
                if filter_class:
                    data['filter_class'] = filter_class
                return self._make_request('POST', '/search/similar', files=files, data=data)
        elif image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            json_data = {
                'image_base64': image_base64,
                'top_k': top_k,
                'metric': metric,
                'filter_class': filter_class
            }
            return self._make_request('POST', '/search/similar', json=json_data)
        else:
            return {'success': False, 'error': 'No image provided'}
    
    def search_by_object(self, image_path, query_image_id=None, top_k=10, metric='cosine', confidence=0.25, aggregation='best_match'):
        """
        Search for images with similar objects.
        
        Args:
            image_path: Query image path
            query_image_id: ID of query image to exclude from results
            top_k: Number of results
            metric: Distance metric
            confidence: Detection confidence
            aggregation: Score aggregation method
            
        Returns:
            API response with results
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'top_k': top_k,
                'metric': metric,
                'confidence': confidence,
                'aggregation': aggregation
            }
            if query_image_id is not None:
                data['query_image_id'] = str(query_image_id)
            return self._make_request('POST', '/search/by-object', files=files, data=data)
    
    def search_by_selected_objects(self, image_path, objects, query_image_id=None, top_k=10, metric='cosine', aggregation='best_match'):
        """
        Search for images with similar objects using specific selected objects.
        
        Args:
            image_path: Query image path
            objects: List of object dicts with class_name and bbox
            query_image_id: ID of query image to exclude from results
            top_k: Number of results
            metric: Distance metric
            aggregation: Score aggregation method
            
        Returns:
            API response with results
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'objects': json.dumps(objects),
                'top_k': top_k,
                'metric': metric,
                'aggregation': aggregation
            }
            if query_image_id is not None:
                data['query_image_id'] = str(query_image_id)
            return self._make_request('POST', '/search/by-selected-objects', files=files, data=data)
    
    # Image transformation endpoints
    
    def transform_image(self, image_path, transform_type, params=None, save=False):
        """
        Apply transformation to an image.
        
        Args:
            image_path: Path to image
            transform_type: 'crop', 'resize', 'rotate', 'flip'
            params: Transform parameters
            save: Save transformed image
            
        Returns:
            API response with transformed image
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = params or {}
            data['save'] = str(save).lower()
            return self._make_request('POST', f'/transform/{transform_type}', files=files, data=data)
    
    def crop_image(self, image_path, x_min, y_min, x_max, y_max, save=False):
        """Crop image to specified region"""
        return self.transform_image(image_path, 'crop', {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        }, save=save)
    
    def resize_image(self, image_path, width=None, height=None, scale=None, save=False):
        """Resize image"""
        params = {}
        if width:
            params['width'] = width
        if height:
            params['height'] = height
        if scale:
            params['scale'] = scale
        return self.transform_image(image_path, 'resize', params, save=save)
    
    def rotate_image(self, image_path, angle, save=False):
        """Rotate image by angle"""
        return self.transform_image(image_path, 'rotate', {'angle': angle}, save=save)
    
    def flip_image(self, image_path, direction='horizontal', save=False):
        """Flip image"""
        return self.transform_image(image_path, 'flip', {'direction': direction}, save=save)
    
    # Index management
    
    def add_to_index(self, image_path, image_id, metadata=None):
        """Add image to similarity search index (legacy endpoint)"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'image_id': image_id}
            if metadata:
                data['metadata'] = json.dumps(metadata)
            return self._make_request('POST', '/search/index', files=files, data=data)
    
    def add_to_object_index(self, image_path, image_id, confidence=0.25):
        """
        Add image to object-based search index.
        This properly detects objects, extracts features, and indexes them.
        
        Args:
            image_path: Path to image file
            image_id: Unique ID for this image
            confidence: Detection confidence threshold
            
        Returns:
            API response with indexed objects and extracted descriptors
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'image_id': str(image_id),
                'confidence': confidence
            }
            return self._make_request('POST', '/index/add', files=files, data=data)
    
    def get_index_stats(self):
        """Get object-based index statistics"""
        return self._make_request('GET', '/index/stats')
    
    def clear_object_index(self):
        """Clear the object-based search index"""
        return self._make_request('DELETE', '/index/clear')
    
    def get_index_info(self):
        """Get similarity index information"""
        return self._make_request('GET', '/search/index')
    
    def clear_index(self):
        """Clear similarity index"""
        return self._make_request('DELETE', '/search/index')
    
    # Health check
    
    def health_check(self):
        """Check if Flask API is running"""
        try:
            response = requests.get(f"{self.base_url.replace('/api', '')}/health", timeout=5)
            return response.json()
        except:
            return {'status': 'unhealthy', 'message': 'Could not connect to Flask API'}
    
    # Generic HTTP methods for 3D endpoints
    
    def get(self, endpoint, params=None):
        """Generic GET request with optional query parameters"""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint, data=None, json_data=None):
        """Generic POST request with JSON data"""
        if data is not None:
            return self._make_request('POST', endpoint, json=data)
        elif json_data is not None:
            return self._make_request('POST', endpoint, json=json_data)
        else:
            return self._make_request('POST', endpoint)
    
    def delete(self, endpoint):
        """Generic DELETE request"""
        return self._make_request('DELETE', endpoint)
    
    def upload_file(self, endpoint, files, data=None):
        """
        Upload file(s) to endpoint.
        
        Args:
            endpoint: API endpoint
            files: Dict of {field_name: (filename, file_content)}
            data: Optional form data
            
        Returns:
            API response
        """
        return self._make_request('POST', endpoint, files=files, data=data)


# Singleton instance
api_client = FlaskAPIClient()
