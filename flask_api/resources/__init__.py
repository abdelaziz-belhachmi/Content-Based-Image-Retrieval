# Flask API Resources
from .detection import DetectionResource, BatchDetectionResource
from .descriptors import (
    DescriptorExtractionResource,
    ObjectDescriptorResource,
    StoredDescriptorResource
)
from .search import SimilaritySearchResource, ObjectSearchResource, IndexManagementResource
from .images import ImageUploadResource, ImageResource, ImageTransformResource

__all__ = [
    'DetectionResource',
    'BatchDetectionResource',
    'DescriptorExtractionResource',
    'ObjectDescriptorResource',
    'StoredDescriptorResource',
    'SimilaritySearchResource',
    'ObjectSearchResource',
    'IndexManagementResource',
    'ImageUploadResource',
    'ImageResource',
    'ImageTransformResource'
]
