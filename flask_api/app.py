"""
Flask REST API - Main Application Entry Point
WITH PERSISTENT OBJECT-BASED SEARCH!
"""

import os
from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from pathlib import Path

from config import get_config


def create_app(config_class=None):
    """
    Application factory for creating Flask app.
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Create upload folder
    upload_folder = Path(app.config.get('UPLOAD_FOLDER', 'uploads'))
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize Flask-RESTful API
    api = Api(app, prefix=app.config.get('API_PREFIX', '/api'))
    
    # Register resources (endpoints)
    register_resources(api)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        from services.similarity_objects import object_similarity_service
        stats = object_similarity_service.get_statistics()
        return {
            'status': 'healthy',
            'message': 'API is running',
            'index_loaded': True,
            'index_stats': stats
        }
    
    return app


def register_resources(api):
    """Register API resources (endpoints)"""
    from resources.detection import DetectionResource, BatchDetectionResource
    from resources.descriptors import (
        DescriptorExtractionResource, 
        ObjectDescriptorResource,
        StoredDescriptorResource
    )
    from resources.search import SimilaritySearchResource, IndexManagementResource
    from resources.search_objects import (
        ObjectBasedSearchResource, 
        SearchBySelectedObjectsResource,
        IndexBuildResource, 
        IndexStatsResource,
        IndexClearResource,
        IndexAddImageResource
    )
    from resources.images import ImageUploadResource, ImageResource, ImageTransformResource
    from resources.models_3d import (
        Model3DUpload,
        Model3DDescriptors,
        Model3DIndex,
        Model3DIndexBuild,
        Model3DSearch,
        Model3DEvaluate,
        Model3DInfo,
        Model3DFile
    )
    
    # Detection endpoints
    api.add_resource(DetectionResource, '/detect')
    api.add_resource(BatchDetectionResource, '/detect/batch')
    
    # Descriptor endpoints
    api.add_resource(DescriptorExtractionResource, '/descriptors/extract')
    api.add_resource(ObjectDescriptorResource, '/descriptors/extract/object')
    api.add_resource(StoredDescriptorResource, '/descriptors/<int:image_id>')
    
    # Object-based search endpoints (PRIMARY)
    api.add_resource(ObjectBasedSearchResource, '/search/by-object')
    api.add_resource(SearchBySelectedObjectsResource, '/search/by-selected-objects')
    api.add_resource(IndexBuildResource, '/index/build')
    api.add_resource(IndexStatsResource, '/index/stats')
    api.add_resource(IndexClearResource, '/index/clear')
    api.add_resource(IndexAddImageResource, '/index/add')
    
    # Legacy search endpoints
    api.add_resource(SimilaritySearchResource, '/search/similar')
    api.add_resource(IndexManagementResource, '/search/index')
    
    # Image management endpoints
    api.add_resource(ImageUploadResource, '/images/upload')
    api.add_resource(ImageResource, '/images/<int:image_id>')
    api.add_resource(ImageTransformResource, '/transform/<string:transform_type>')
    
    # 3D Model retrieval endpoints
    api.add_resource(Model3DUpload, '/models3d/upload')
    api.add_resource(Model3DDescriptors, '/models3d/descriptors')
    api.add_resource(Model3DIndex, '/models3d/index')
    api.add_resource(Model3DIndexBuild, '/models3d/index/build')
    api.add_resource(Model3DSearch, '/models3d/search')
    api.add_resource(Model3DEvaluate, '/models3d/evaluate')
    api.add_resource(Model3DInfo, '/models3d/<string:model_id>')
    api.add_resource(Model3DFile, '/models3d/<string:model_id>/file')


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        return {
            'success': False,
            'error': 'Bad Request',
            'message': str(error.description)
        }, 400
    
    @app.errorhandler(404)
    def not_found(error):
        return {
            'success': False,
            'error': 'Not Found',
            'message': 'The requested resource was not found'
        }, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {
            'success': False,
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }, 500


# Application instance
app = create_app()


if __name__ == '__main__':
    print("=" * 70)
    print("Flask REST API for Object-Based Image Search + 3D Model Retrieval")
    print("=" * 70)
    print("KEY ENDPOINTS:")
    print("  POST /api/index/build         - Build persistent search index")
    print("  GET  /api/index/stats         - View index statistics")
    print("  POST /api/search/by-object    - Search by object classes")
    print("  DELETE /api/index/clear       - Clear the index")
    print("")
    print("3D MODEL ENDPOINTS:")
    print("  POST /api/models3d/upload     - Upload 3D model (OBJ)")
    print("  POST /api/models3d/descriptors - Extract 3D descriptors")
    print("  POST /api/models3d/index      - Add model to 3D index")
    print("  POST /api/models3d/index/build - Batch index 3D models")
    print("  POST /api/models3d/search     - Search similar 3D models")
    print("  GET  /api/models3d/<id>       - Get 3D model info")
    print("")
    print("OTHER ENDPOINTS:")
    print("  POST /api/detect              - Detect objects in image")
    print("  POST /api/descriptors/extract - Extract descriptors")
    print("  POST /api/images/upload       - Upload images")
    print("  GET  /health                  - Health check")
    print("=" * 70)
    print("FEATURES:")
    print("  ✓ Persistent index (survives restarts)")
    print("  ✓ Object-class filtering (bus finds only bus)")
    print("  ✓ Automatic saving on every index operation")
    print("  ✓ 3D Local Features: Spin Images, Shape Contexts, PFH")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)