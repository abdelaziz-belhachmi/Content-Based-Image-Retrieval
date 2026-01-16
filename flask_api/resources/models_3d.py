"""
Flask API Resources for 3D Model Retrieval

Provides REST endpoints for 3D model upload, indexing, and similarity search.
"""

import os
import uuid
from pathlib import Path
from flask import request, send_file, current_app
from flask_restful import Resource
from werkzeug.utils import secure_filename

from services.mesh_loader import load_and_normalize, OBJLoader
from services.descriptors_3d import extract_descriptors, extract_combined_descriptor
from services.similarity_3d import get_similarity_service


# Allowed file extensions
ALLOWED_EXTENSIONS = {'obj'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_upload_folder() -> Path:
    """Get the 3D models upload folder."""
    folder = Path(current_app.config.get('UPLOAD_FOLDER_3D', 'uploads_3d'))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class Model3DUpload(Resource):
    """Upload a 3D model file."""
    
    def post(self):
        """
        Upload a 3D model (OBJ format).
        
        Request:
            - file: The OBJ file (multipart/form-data)
            - category: Optional category/class name
            
        Returns:
            - model_id: Unique identifier
            - filepath: Path to uploaded file
            - mesh_stats: Mesh statistics
        """
        if 'file' not in request.files:
            return {'success': False, 'error': 'No file provided'}, 400
        
        file = request.files['file']
        
        if file.filename == '':
            return {'success': False, 'error': 'No file selected'}, 400
        
        if not allowed_file(file.filename):
            return {'success': False, 'error': 'Only OBJ files are allowed'}, 400
        
        try:
            # Generate unique ID and save file
            model_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            
            upload_folder = get_upload_folder()
            filepath = upload_folder / f"{model_id}_{filename}"
            file.save(str(filepath))
            
            # Load mesh to get stats
            mesh = load_and_normalize(str(filepath))
            stats = mesh.get_stats()
            
            # Get category if provided
            category = request.form.get('category', None)
            
            return {
                'success': True,
                'model_id': model_id,
                'filename': filename,
                'filepath': str(filepath),
                'category': category,
                'mesh_stats': stats
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500


class Model3DDescriptors(Resource):
    """Extract descriptors from a 3D model."""
    
    def post(self):
        """
        Extract local feature descriptors from a 3D model.
        
        Request JSON:
            - filepath: Path to the OBJ file
            - descriptor_types: Optional list of descriptor types
              ['spin_image', 'shape_context', 'shape_index', 'pfh']
            
        Returns:
            - descriptors: Dictionary of descriptor arrays
            - combined_size: Size of combined vector
        """
        data = request.get_json() or {}
        filepath = data.get('filepath')
        
        if not filepath:
            # Try to get from file upload
            if 'file' in request.files:
                file = request.files['file']
                if file and allowed_file(file.filename):
                    # Save temporarily
                    upload_folder = get_upload_folder()
                    temp_path = upload_folder / f"temp_{uuid.uuid4()}.obj"
                    file.save(str(temp_path))
                    filepath = str(temp_path)
        
        if not filepath:
            return {'success': False, 'error': 'No filepath or file provided'}, 400
        
        if not Path(filepath).exists():
            return {'success': False, 'error': 'File not found'}, 404
        
        try:
            # Load and normalize mesh
            mesh = load_and_normalize(filepath)
            
            # Extract descriptors
            descriptor_types = data.get('descriptor_types', None)
            descriptors = extract_descriptors(mesh, descriptor_types)
            combined = extract_combined_descriptor(mesh)
            
            # Convert to serializable format
            result = {
                'success': True,
                'descriptors': {
                    name: {
                        'values': vec.tolist(),
                        'size': len(vec)
                    }
                    for name, vec in descriptors.items()
                },
                'combined_vector': {
                    'size': len(combined),
                    'values': combined.tolist()
                },
                'mesh_stats': mesh.get_stats()
            }
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500


class Model3DIndex(Resource):
    """Manage the 3D model index."""
    
    def post(self):
        """
        Add a model to the search index.
        
        Request JSON:
            - model_id: Unique identifier (optional, will be generated)
            - filepath: Path to the OBJ file
            - category: Category/class name
            - metadata: Additional metadata (optional)
            
        Returns:
            - Indexing result
        """
        data = request.get_json() or {}
        
        filepath = data.get('filepath')
        if not filepath:
            return {'success': False, 'error': 'No filepath provided'}, 400
        
        model_id = data.get('model_id', str(uuid.uuid4()))
        category = data.get('category')
        metadata = data.get('metadata')
        
        service = get_similarity_service()
        result = service.index_model(model_id, filepath, category, metadata)
        
        if result.get('success'):
            return result
        else:
            return result, 500
    
    def get(self):
        """
        Get index statistics.
        
        Returns:
            - Index statistics
        """
        service = get_similarity_service()
        return {
            'success': True,
            'stats': service.get_stats()
        }
    
    def delete(self):
        """
        Clear the entire index.
        
        Returns:
            - Success status
        """
        service = get_similarity_service()
        service.index.clear()
        return {
            'success': True,
            'message': 'Index cleared'
        }


class Model3DIndexBuild(Resource):
    """Build index from multiple models."""
    
    def post(self):
        """
        Index multiple 3D models at once.
        
        Request JSON:
            - models: List of {id, filepath, category}
            - directory: Or path to directory with OBJ files
            - category: Default category for directory indexing
            
        Returns:
            - Batch indexing results
        """
        data = request.get_json() or {}
        service = get_similarity_service()
        
        models = data.get('models', [])
        
        # If directory is provided, scan for OBJ files
        directory = data.get('directory')
        if directory:
            default_category = data.get('category', 'unknown')
            dir_path = Path(directory)
            
            if dir_path.exists() and dir_path.is_dir():
                for obj_file in dir_path.glob('**/*.obj'):
                    models.append({
                        'id': obj_file.stem,
                        'filepath': str(obj_file),
                        'category': default_category
                    })
        
        if not models:
            return {'success': False, 'error': 'No models to index'}, 400
        
        result = service.batch_index(models)
        return result


class Model3DSearch(Resource):
    """Search for similar 3D models."""
    
    def post(self):
        """
        Search for similar 3D models.
        
        Request JSON:
            - filepath: Path to query OBJ file
            - model_id: Or ID of indexed model to use as query
            - metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            - k: Number of results (default: 10)
            - category_filter: Filter by category (optional)
            
        Returns:
            - results: List of similar models with distances
        """
        data = request.get_json() or {}
        service = get_similarity_service()
        
        filepath = data.get('filepath')
        model_id = data.get('model_id')
        metric = data.get('metric', 'cosine')
        k = data.get('k', 10)
        category_filter = data.get('category_filter')
        
        try:
            if model_id:
                # Search by indexed model ID
                results = service.search_by_id(
                    model_id, 
                    metric=metric, 
                    k=k
                )
            elif filepath:
                # Search by file path
                results = service.search(
                    query_filepath=filepath,
                    metric=metric,
                    k=k,
                    category_filter=category_filter
                )
            elif 'file' in request.files:
                # Search by uploaded file
                file = request.files['file']
                if file and allowed_file(file.filename):
                    upload_folder = get_upload_folder()
                    temp_path = upload_folder / f"query_{uuid.uuid4()}.obj"
                    file.save(str(temp_path))
                    
                    results = service.search(
                        query_filepath=str(temp_path),
                        metric=metric,
                        k=k,
                        category_filter=category_filter
                    )
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    return {'success': False, 'error': 'Invalid file'}, 400
            else:
                return {'success': False, 'error': 'No query provided'}, 400
            
            return {
                'success': True,
                'query': {
                    'model_id': model_id,
                    'filepath': filepath,
                    'metric': metric,
                    'k': k
                },
                'results': results,
                'count': len(results)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500


class Model3DEvaluate(Resource):
    """Evaluate search performance."""
    
    def post(self):
        """
        Evaluate search performance using standard IR metrics.
        
        Request JSON:
            - filepath: Path to query OBJ file
            - category: Ground truth category
            - metric: Distance metric
            
        Returns:
            - Precision@K, NDCG@K, Average Precision
        """
        data = request.get_json() or {}
        service = get_similarity_service()
        
        filepath = data.get('filepath')
        category = data.get('category')
        metric = data.get('metric', 'cosine')
        
        if not filepath or not category:
            return {'success': False, 'error': 'filepath and category required'}, 400
        
        try:
            mesh = load_and_normalize(filepath)
            query_vector = extract_combined_descriptor(mesh)
            
            metrics = service.evaluate_search(
                query_vector=query_vector,
                query_category=category,
                metric=metric
            )
            
            return {
                'success': True,
                'metrics': metrics
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500


class Model3DInfo(Resource):
    """Get information about a specific model."""
    
    def get(self, model_id: str):
        """
        Get model information by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            - Model info including descriptors
        """
        service = get_similarity_service()
        model = service.index.get_model(model_id)
        
        if not model:
            return {'success': False, 'error': 'Model not found'}, 404
        
        # Don't return full vectors, just sizes
        result = {
            'success': True,
            'model_id': model_id,
            'filepath': model.get('filepath'),
            'category': model.get('category'),
            'metadata': model.get('metadata'),
            'indexed_at': model.get('indexed_at'),
            'has_descriptors': 'descriptors' in model,
            'has_combined_vector': 'combined_vector' in model
        }
        
        if 'descriptors' in model:
            result['descriptor_sizes'] = {
                name: len(vec) for name, vec in model['descriptors'].items()
            }
        
        if 'combined_vector' in model:
            result['combined_vector_size'] = len(model['combined_vector'])
        
        return result
    
    def delete(self, model_id: str):
        """
        Delete a model from the index.
        
        Args:
            model_id: Model identifier
            
        Returns:
            - Success status
        """
        service = get_similarity_service()
        
        if service.index.remove_model(model_id):
            return {'success': True, 'message': f'Model {model_id} removed'}
        else:
            return {'success': False, 'error': 'Model not found'}, 404


class Model3DFile(Resource):
    """Serve 3D model files."""
    
    def get(self, model_id: str):
        """
        Get the OBJ file for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            - OBJ file as attachment
        """
        service = get_similarity_service()
        model = service.index.get_model(model_id)
        
        if not model:
            return {'success': False, 'error': 'Model not found'}, 404
        
        filepath = model.get('filepath')
        if not filepath or not Path(filepath).exists():
            return {'success': False, 'error': 'File not found'}, 404
        
        return send_file(
            filepath,
            mimetype='model/obj',
            as_attachment=True,
            download_name=Path(filepath).name
        )
