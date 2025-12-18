"""
Image Management and Transformation API Resources
"""

import os
import cv2
import numpy as np
from flask import request, current_app, send_file
from flask_restful import Resource
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import Path
import base64
import io


def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def get_upload_folder():
    """Get upload folder path"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    Path(upload_folder).mkdir(parents=True, exist_ok=True)
    return upload_folder


class ImageUploadResource(Resource):
    """
    POST /api/images/upload
    Upload images to the server
    """
    
    def post(self):
        """
        Upload one or more images.
        
        Returns uploaded file info with generated IDs.
        """
        try:
            if 'images' not in request.files and 'image' not in request.files:
                return {'success': False, 'message': 'No images provided'}, 400
            
            # Get files (support both 'image' and 'images')
            files = request.files.getlist('images')
            if not files or files[0].filename == '':
                files = request.files.getlist('image')
            
            upload_folder = get_upload_folder()
            uploaded = []
            errors = []
            
            for file in files:
                if file.filename == '':
                    continue
                
                if not allowed_file(file.filename):
                    errors.append({
                        'filename': file.filename,
                        'error': 'File type not allowed'
                    })
                    continue
                
                try:
                    # Secure the filename
                    filename = secure_filename(file.filename)
                    
                    # Generate unique filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    name, ext = os.path.splitext(filename)
                    unique_filename = f"{name}_{timestamp}{ext}"
                    
                    # Save file
                    filepath = os.path.join(upload_folder, unique_filename)
                    file.save(filepath)
                    
                    # Get image info
                    image = cv2.imread(filepath)
                    if image is not None:
                        height, width = image.shape[:2]
                    else:
                        height, width = 0, 0
                    
                    uploaded.append({
                        'original_filename': file.filename,
                        'saved_filename': unique_filename,
                        'filepath': filepath,
                        'size': os.path.getsize(filepath),
                        'width': width,
                        'height': height
                    })
                    
                except Exception as e:
                    errors.append({
                        'filename': file.filename,
                        'error': str(e)
                    })
            
            return {
                'success': len(uploaded) > 0,
                'data': {
                    'uploaded': uploaded,
                    'errors': errors,
                    'total_uploaded': len(uploaded),
                    'total_errors': len(errors)
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200 if len(uploaded) > 0 else 400
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class ImageResource(Resource):
    """
    GET/DELETE /api/images/<image_id>
    Get or delete an image
    """
    
    def get(self, image_id):
        """
        Get image by ID (filename).
        """
        try:
            upload_folder = get_upload_folder()
            
            # Try to find the image
            for ext in ['', '.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                filepath = os.path.join(upload_folder, f"{image_id}{ext}")
                if os.path.exists(filepath):
                    return send_file(filepath)
            
            return {
                'success': False,
                'message': f'Image not found: {image_id}'
            }, 404
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500
    
    def delete(self, image_id):
        """
        Delete image by ID (filename).
        """
        try:
            upload_folder = get_upload_folder()
            
            # Try to find and delete the image
            for ext in ['', '.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                filepath = os.path.join(upload_folder, f"{image_id}{ext}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return {
                        'success': True,
                        'message': f'Image deleted: {image_id}',
                        'timestamp': datetime.utcnow().isoformat() + 'Z'
                    }, 200
            
            return {
                'success': False,
                'message': f'Image not found: {image_id}'
            }, 404
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500


class ImageTransformResource(Resource):
    """
    POST /api/transform/<transform_type>
    Apply transformations to images
    """
    
    def post(self, transform_type):
        """
        Apply transformation to an image.
        
        Supported transforms:
            - crop: Crop image to region
            - resize: Resize image
            - rotate: Rotate image
            - flip: Flip image horizontally/vertically
        """
        try:
            # Get image
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
            
            if image is None:
                return {'success': False, 'message': 'No valid image provided'}, 400
            
            # Get parameters
            params = request.json if request.is_json else request.form
            
            # Apply transformation
            if transform_type == 'crop':
                result = self._crop(image, params)
            elif transform_type == 'resize':
                result = self._resize(image, params)
            elif transform_type == 'rotate':
                result = self._rotate(image, params)
            elif transform_type == 'flip':
                result = self._flip(image, params)
            else:
                return {
                    'success': False,
                    'message': f'Unknown transform type: {transform_type}'
                }, 400
            
            # Encode result as base64
            _, buffer = cv2.imencode('.jpg', result)
            result_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Optionally save
            save = params.get('save', 'false')
            saved_path = None
            
            if str(save).lower() == 'true':
                upload_folder = get_upload_folder()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"transformed_{timestamp}.jpg"
                saved_path = os.path.join(upload_folder, filename)
                cv2.imwrite(saved_path, result)
            
            return {
                'success': True,
                'data': {
                    'transform_type': transform_type,
                    'original_size': {
                        'width': image.shape[1],
                        'height': image.shape[0]
                    },
                    'result_size': {
                        'width': result.shape[1],
                        'height': result.shape[0]
                    },
                    'result_image': f"data:image/jpeg;base64,{result_base64}",
                    'saved_path': saved_path
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, 500
    
    def _crop(self, image, params):
        """Crop image to specified region"""
        x_min = int(params.get('x_min', 0))
        y_min = int(params.get('y_min', 0))
        x_max = int(params.get('x_max', image.shape[1]))
        y_max = int(params.get('y_max', image.shape[0]))
        
        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)
        
        return image[y_min:y_max, x_min:x_max]
    
    def _resize(self, image, params):
        """Resize image to specified dimensions"""
        width = params.get('width')
        height = params.get('height')
        scale = params.get('scale')
        
        if scale:
            scale = float(scale)
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
        elif width and height:
            width = int(width)
            height = int(height)
        elif width:
            width = int(width)
            ratio = width / image.shape[1]
            height = int(image.shape[0] * ratio)
        elif height:
            height = int(height)
            ratio = height / image.shape[0]
            width = int(image.shape[1] * ratio)
        else:
            return image
        
        return cv2.resize(image, (width, height))
    
    def _rotate(self, image, params):
        """Rotate image by specified angle"""
        angle = float(params.get('angle', 0))
        
        # Get image center
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounds
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust matrix
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        
        return cv2.warpAffine(image, matrix, (new_w, new_h))
    
    def _flip(self, image, params):
        """Flip image horizontally or vertically"""
        direction = params.get('direction', 'horizontal')
        
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        elif direction == 'both':
            return cv2.flip(image, -1)
        else:
            return image
