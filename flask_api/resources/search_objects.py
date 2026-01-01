"""
Enhanced Similarity Search API - Object-Based Search
NOW WITH PERSISTENT INDEX!
"""

import cv2
import numpy as np
from flask import request
from flask_restful import Resource
from datetime import datetime
import base64

from services.descriptors import feature_extractor
from services.similarity_objects import object_similarity_service
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


class ObjectBasedSearchResource(Resource):
    """
    POST /api/search/by-object
    Search for images containing ONLY the same object classes as the query image.
    
    KEY FEATURE: If query image has [bus, person], results will ONLY contain
    images with bus OR person OR both. NO other objects like bananas!
    """
    
    def post(self):
        """
        Find images containing the same objects as the query image.
        
        Accepts:
            - image: Query image (file or base64)
            - query_image_id: Optional ID to exclude from results
            - top_k: Number of result images (default 10)
            - metric: Distance metric (default 'cosine')
            - confidence: Detection confidence threshold (default 0.25)
            - aggregation: Score aggregation method (default 'best_match')
        
        Returns:
            Images ranked by similarity, containing ONLY matching object classes
        """
        try:
            image = decode_image(request)
            
            if image is None:
                return {'success': False, 'message': 'No valid image provided'}, 400
            
            # Get parameters
            if request.is_json:
                top_k = int(request.json.get('top_k', 10))
                metric = request.json.get('metric', 'cosine')
                confidence = float(request.json.get('confidence', 0.25))
                aggregation = request.json.get('aggregation', 'best_match')
                query_image_id = request.json.get('query_image_id')
            else:
                top_k = int(request.form.get('top_k', 10))
                metric = request.form.get('metric', 'cosine')
                confidence = float(request.form.get('confidence', 0.25))
                aggregation = request.form.get('aggregation', 'best_match')
                query_image_id = request.form.get('query_image_id')
            
            # Validate parameters
            valid_aggregations = ['best_match', 'average', 'min_distance', 'any_match']
            if aggregation not in valid_aggregations:
                return {
                    'success': False,
                    'message': f'Invalid aggregation. Must be one of: {valid_aggregations}'
                }, 400
            
            if metric not in object_similarity_service.SUPPORTED_METRICS:
                return {
                    'success': False,
                    'message': f'Invalid metric. Supported: {object_similarity_service.SUPPORTED_METRICS}'
                }, 400
            
            # Check if index is empty
            if object_similarity_service.get_index_size() == 0:
                return {
                    'success': True,
                    'data': {
                        'message': 'Index is empty. Please build the index first using /api/index/build',
                        'index_size': 0,
                        'results': []
                    },
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }, 200
            
            # Step 1: Detect objects in query image
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
            
            # Step 2: Extract features for each detected object
            query_objects = []
            
            for detection in detections:
                bbox = detection['bbox']
                
                # Crop object region
                cropped = image[
                    bbox['y_min']:bbox['y_max'],
                    bbox['x_min']:bbox['x_max']
                ]
                
                if cropped.size == 0:
                    continue
                
                # Extract features
                feature_vector = feature_extractor.extract_feature_vector(cropped)
                
                query_objects.append({
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'bbox': bbox,
                    'feature_vector': feature_vector
                })
            
            if not query_objects:
                return {
                    'success': True,
                    'data': {
                        'message': 'Could not extract features from detected objects',
                        'results': []
                    },
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }, 200
            
            # Step 3: Search for images with matching objects
            results = object_similarity_service.search_by_image_objects(
                query_objects=query_objects,
                top_k=top_k,
                metric=metric,
                aggregation=aggregation,
                exclude_image_id=query_image_id
            )
            
            # Get query class names
            query_classes = list(set([obj['class_name'] for obj in query_objects]))
            
            return {
                'success': True,
                'data': {
                    'query_image_size': {
                        'width': image.shape[1],
                        'height': image.shape[0]
                    },
                    'num_query_objects': len(query_objects),
                    'query_classes': query_classes,
                    'query_detections': detections,
                    'search_config': {
                        'metric': metric,
                        'aggregation': aggregation,
                        'top_k': top_k
                    },
                    'index_stats': object_similarity_service.get_statistics(),
                    'num_results': len(results),
                    'results': results,
                    'explanation': f"Found {len(results)} images containing: {', '.join(query_classes)}"
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class IndexBuildResource(Resource):
    """
    POST /api/index/build
    Build the object-based search index from a dataset.
    INDEX IS AUTOMATICALLY SAVED AND PERSISTS ACROSS RESTARTS!
    """
    
    def post(self):
        """
        Build search index by detecting and indexing objects from uploaded images.
        
        Accepts:
            - images: List of images (multipart/form-data)
            - confidence: Detection confidence threshold (default 0.25)
        
        This processes each image, detects objects, extracts features,
        and adds them to the searchable index WITH AUTOMATIC PERSISTENCE.
        """
        try:
            if 'images' not in request.files:
                return {
                    'success': False,
                    'message': 'No images provided. Send images as multipart/form-data'
                }, 400
            
            files = request.files.getlist('images')
            confidence = float(request.form.get('confidence', 0.25))
            
            indexed_count = 0
            errors = []
            indexed_images = []
            
            for idx, file in enumerate(files):
                try:
                    if file.filename == '':
                        continue
                    
                    # Read image
                    file_bytes = file.read()
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        errors.append({
                            'filename': file.filename,
                            'error': 'Could not decode image'
                        })
                        continue
                    
                    # Detect objects
                    detections = detection_service.detect(image, confidence_threshold=confidence)
                    
                    if not detections:
                        errors.append({
                            'filename': file.filename,
                            'error': 'No objects detected'
                        })
                        continue
                    
                    # Extract and index each object
                    image_id = f"img_{idx}_{file.filename}"
                    object_count = 0
                    
                    for obj_idx, detection in enumerate(detections):
                        bbox = detection['bbox']
                        
                        # Crop object
                        cropped = image[
                            bbox['y_min']:bbox['y_max'],
                            bbox['x_min']:bbox['x_max']
                        ]
                        
                        if cropped.size == 0:
                            continue
                        
                        # Extract features
                        feature_vector = feature_extractor.extract_feature_vector(cropped)
                        
                        # Create object ID
                        object_id = f"{image_id}_obj_{obj_idx}"
                        
                        # Index object (automatically persisted)
                        object_similarity_service.add_object_to_index(
                            object_id=object_id,
                            feature_vector=feature_vector,
                            metadata={
                                'image_id': image_id,
                                'class_name': detection['class_name'],
                                'confidence': detection['confidence'],
                                'bbox': bbox,
                                'filename': file.filename
                            }
                        )
                        
                        indexed_count += 1
                        object_count += 1
                    
                    indexed_images.append({
                        'filename': file.filename,
                        'image_id': image_id,
                        'objects_indexed': object_count
                    })
                
                except Exception as e:
                    errors.append({
                        'filename': file.filename,
                        'error': str(e)
                    })
            
            stats = object_similarity_service.get_statistics()
            
            return {
                'success': True,
                'data': {
                    'indexed_objects': indexed_count,
                    'indexed_images': len(indexed_images),
                    'images': indexed_images,
                    'errors': errors,
                    'index_stats': stats,
                    'message': f'Successfully indexed {indexed_count} objects from {len(indexed_images)} images. Index is saved and will persist across restarts.'
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
        
        except Exception as e:
            import traceback
            return {
                'success': False,
                'message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class IndexStatsResource(Resource):
    """
    GET /api/index/stats
    Get statistics about the search index.
    """
    
    def get(self):
        """Get index statistics"""
        try:
            stats = object_similarity_service.get_statistics()
            
            return {
                'success': True,
                'data': {
                    **stats,
                    'persistence': 'enabled',
                    'message': 'Index is automatically saved and persists across restarts'
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
        
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class IndexClearResource(Resource):
    """
    DELETE /api/index/clear
    Clear the entire search index.
    """
    
    def delete(self):
        """Clear the index"""
        try:
            object_similarity_service.clear_index()
            
            return {
                'success': True,
                'message': 'Index cleared successfully',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
        
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class IndexAddImageResource(Resource):
    """
    POST /api/index/add
    Add a single image to the object-based search index.
    This is used when uploading images to automatically index them.
    """
    
    def post(self):
        """
        Add a single image to the index by detecting objects and extracting features.
        
        Accepts:
            - image: Image file (multipart/form-data)
            - image_id: Unique ID for this image (required)
            - confidence: Detection confidence threshold (default 0.25)
        
        Returns:
            Indexed objects info and extracted descriptors
        """
        import traceback
        
        try:
            image = decode_image(request)
            
            if image is None:
                return {
                    'success': False,
                    'message': 'No valid image provided'
                }, 400
            
            # Get image_id from form data
            image_id = request.form.get('image_id')
            if not image_id:
                return {
                    'success': False,
                    'message': 'image_id is required'
                }, 400
            
            confidence = float(request.form.get('confidence', 0.25))
            
            # Detect objects in the image
            detections = detection_service.detect(image, confidence_threshold=confidence)
            
            if not detections:
                return {
                    'success': True,
                    'data': {
                        'image_id': image_id,
                        'indexed_objects': 0,
                        'detections': [],
                        'descriptors': [],
                        'message': 'No objects detected in image'
                    },
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }, 200
            
            # Extract and index each detected object
            indexed_objects = []
            descriptors_list = []
            
            for obj_idx, detection in enumerate(detections):
                bbox = detection['bbox']
                
                # Crop object region
                cropped = image[
                    bbox['y_min']:bbox['y_max'],
                    bbox['x_min']:bbox['x_max']
                ]
                
                if cropped.size == 0:
                    continue
                
                # Extract feature vector (combined descriptor)
                feature_vector = feature_extractor.extract_feature_vector(cropped)
                
                # Extract detailed descriptors for storage
                descriptors = feature_extractor.extract_all_features(cropped)
                
                # Create unique object ID
                object_id = f"{image_id}_obj_{obj_idx}"
                
                # Add to index (automatically persisted)
                object_similarity_service.add_object_to_index(
                    object_id=object_id,
                    feature_vector=feature_vector,
                    metadata={
                        'image_id': image_id,
                        'class_name': detection['class_name'],
                        'class_id': detection['class_id'],
                        'confidence': detection['confidence'],
                        'bbox': bbox
                    }
                )
                
                indexed_objects.append({
                    'object_id': object_id,
                    'class_name': detection['class_name'],
                    'class_id': detection['class_id'],
                    'confidence': detection['confidence'],
                    'bbox': bbox
                })
                
                # Prepare descriptors for response
                descriptors_list.append({
                    'object_id': object_id,
                    'class_name': detection['class_name'],
                    'class_id': detection['class_id'],
                    'bbox': bbox,
                    'descriptors': {
                        'dominant_colors': descriptors.get('dominant_colors'),
                        'color_moments': descriptors.get('color_moments'),
                        'tamura': descriptors.get('tamura'),
                        'glcm': descriptors.get('glcm'),
                        'hu_moments': descriptors.get('hu_moments'),
                        'contour': descriptors.get('contour'),
                    },
                    'feature_vector_length': len(feature_vector)
                })
            
            stats = object_similarity_service.get_statistics()
            
            return {
                'success': True,
                'data': {
                    'image_id': image_id,
                    'indexed_objects': len(indexed_objects),
                    'objects': indexed_objects,
                    'descriptors': descriptors_list,
                    'index_stats': stats,
                    'message': f'Successfully indexed {len(indexed_objects)} objects'
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500