"""
3D Model Similarity Search Service

Provides indexing and similarity search functionality for 3D models
using local feature-based descriptors.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.distance import cdist
import logging
import threading
from datetime import datetime

from .mesh_loader import Mesh3D, load_and_normalize
from .descriptors_3d import extract_combined_descriptor, extract_descriptors

logger = logging.getLogger(__name__)


class Model3DIndex:
    """
    Index structure for 3D model retrieval.
    
    Stores descriptors and metadata for efficient similarity search.
    """
    
    def __init__(self, index_path: str = None):
        """
        Initialize the 3D model index.
        
        Args:
            index_path: Path to save/load the index
        """
        self.index_path = index_path or 'model3d_index.json'
        self.models: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self) -> None:
        """Load index from disk."""
        path = Path(self.index_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.models = data.get('models', {})
                    logger.info(f"Loaded 3D index with {len(self.models)} models")
            except Exception as e:
                logger.warning(f"Failed to load 3D index: {e}")
                self.models = {}
    
    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            with open(self.index_path, 'w') as f:
                json.dump({
                    'models': self.models,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save 3D index: {e}")
    
    def add_model(self, 
                  model_id: str, 
                  filepath: str,
                  category: str = None,
                  descriptors: Dict[str, np.ndarray] = None,
                  combined_vector: np.ndarray = None,
                  metadata: Dict[str, Any] = None) -> bool:
        """
        Add a 3D model to the index.
        
        Args:
            model_id: Unique identifier for the model
            filepath: Path to the OBJ file
            category: Category/class of the model
            descriptors: Dictionary of individual descriptors
            combined_vector: Combined feature vector
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        with self._lock:
            entry = {
                'filepath': filepath,
                'category': category,
                'metadata': metadata or {},
                'indexed_at': datetime.now().isoformat()
            }
            
            # Store descriptors as lists for JSON serialization
            if descriptors:
                entry['descriptors'] = {
                    name: vec.tolist() for name, vec in descriptors.items()
                }
            
            if combined_vector is not None:
                entry['combined_vector'] = combined_vector.tolist()
            
            self.models[model_id] = entry
            self._save_index()
            
            return True
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the index."""
        with self._lock:
            if model_id in self.models:
                del self.models[model_id]
                self._save_index()
                return True
            return False
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model entry by ID."""
        return self.models.get(model_id)
    
    def get_all_model_ids(self) -> List[str]:
        """Get all model IDs in the index."""
        return list(self.models.keys())
    
    def get_models_by_category(self, category: str) -> List[str]:
        """Get model IDs by category."""
        return [
            mid for mid, entry in self.models.items()
            if entry.get('category') == category
        ]
    
    def get_all_vectors(self) -> Tuple[List[str], np.ndarray]:
        """
        Get all combined vectors for similarity computation.
        
        Returns:
            Tuple of (model_ids, vectors_matrix)
        """
        model_ids = []
        vectors = []
        
        for mid, entry in self.models.items():
            if 'combined_vector' in entry:
                model_ids.append(mid)
                vectors.append(entry['combined_vector'])
        
        if vectors:
            return model_ids, np.array(vectors)
        return [], np.array([])
    
    def clear(self) -> None:
        """Clear the entire index."""
        with self._lock:
            self.models = {}
            self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        categories = {}
        for entry in self.models.values():
            cat = entry.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_models': len(self.models),
            'categories': categories,
            'index_path': self.index_path
        }


class Similarity3DService:
    """
    Service for 3D model similarity search.
    """
    
    DISTANCE_METRICS = ['euclidean', 'cosine', 'manhattan', 'correlation']
    
    def __init__(self, index: Model3DIndex = None):
        """
        Initialize the similarity service.
        
        Args:
            index: Model3DIndex instance
        """
        self.index = index or Model3DIndex()
    
    def index_model(self, 
                    model_id: str,
                    filepath: str,
                    category: str = None,
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Index a 3D model from file.
        
        Args:
            model_id: Unique identifier
            filepath: Path to OBJ file
            category: Model category
            metadata: Additional metadata
            
        Returns:
            Indexing result with descriptors info
        """
        try:
            # Load and normalize mesh
            mesh = load_and_normalize(filepath)
            
            # Extract descriptors
            descriptors = extract_descriptors(mesh)
            combined = extract_combined_descriptor(mesh)
            
            # Get mesh stats
            stats = mesh.get_stats()
            if metadata is None:
                metadata = {}
            metadata['mesh_stats'] = stats
            
            # Add to index
            self.index.add_model(
                model_id=model_id,
                filepath=filepath,
                category=category,
                descriptors=descriptors,
                combined_vector=combined,
                metadata=metadata
            )
            
            return {
                'success': True,
                'model_id': model_id,
                'category': category,
                'descriptor_sizes': {name: len(vec) for name, vec in descriptors.items()},
                'combined_size': len(combined),
                'mesh_stats': stats
            }
            
        except Exception as e:
            logger.error(f"Failed to index model {model_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search(self,
               query_filepath: str = None,
               query_vector: np.ndarray = None,
               metric: str = 'cosine',
               k: int = 10,
               category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar 3D models.
        
        Args:
            query_filepath: Path to query OBJ file
            query_vector: Pre-computed query vector (alternative to filepath)
            metric: Distance metric to use
            k: Number of results to return
            category_filter: Filter results by category
            
        Returns:
            List of search results with model info and distances
        """
        # Get query vector
        if query_vector is None:
            if query_filepath is None:
                raise ValueError("Either query_filepath or query_vector must be provided")
            
            mesh = load_and_normalize(query_filepath)
            query_vector = extract_combined_descriptor(mesh)
        
        # Get indexed vectors
        model_ids, vectors = self.index.get_all_vectors()
        
        if len(model_ids) == 0:
            return []
        
        # Apply category filter if specified
        if category_filter:
            filtered_ids = self.index.get_models_by_category(category_filter)
            mask = [mid in filtered_ids for mid in model_ids]
            model_ids = [mid for mid, m in zip(model_ids, mask) if m]
            vectors = vectors[mask]
        
        if len(model_ids) == 0:
            return []
        
        # Compute distances
        query_vector = query_vector.reshape(1, -1)
        
        if metric == 'cosine':
            # Normalize for cosine distance
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
            distances = 1 - np.dot(vectors_norm, query_norm.T).flatten()
        else:
            distances = cdist(query_vector, vectors, metric=metric).flatten()
        
        # Sort by distance
        sorted_indices = np.argsort(distances)[:k]
        
        # Build results
        results = []
        for idx in sorted_indices:
            model_id = model_ids[idx]
            model_entry = self.index.get_model(model_id)
            
            results.append({
                'model_id': model_id,
                'distance': float(distances[idx]),
                'similarity': float(1.0 / (1.0 + distances[idx])),
                'filepath': model_entry.get('filepath'),
                'category': model_entry.get('category'),
                'metadata': model_entry.get('metadata', {})
            })
        
        return results
    
    def search_by_id(self,
                     query_model_id: str,
                     metric: str = 'cosine',
                     k: int = 10,
                     exclude_self: bool = True) -> List[Dict[str, Any]]:
        """
        Search for models similar to an indexed model.
        
        Args:
            query_model_id: ID of the query model in the index
            metric: Distance metric
            k: Number of results
            exclude_self: Whether to exclude the query model from results
            
        Returns:
            List of search results
        """
        model_entry = self.index.get_model(query_model_id)
        if not model_entry or 'combined_vector' not in model_entry:
            return []
        
        query_vector = np.array(model_entry['combined_vector'])
        
        results = self.search(
            query_vector=query_vector,
            metric=metric,
            k=k + (1 if exclude_self else 0)
        )
        
        if exclude_self:
            results = [r for r in results if r['model_id'] != query_model_id][:k]
        
        return results
    
    def batch_index(self, 
                    models: List[Dict[str, str]],
                    category: str = None) -> Dict[str, Any]:
        """
        Index multiple models.
        
        Args:
            models: List of {'id': ..., 'filepath': ..., 'category': ...}
            category: Default category if not specified per model
            
        Returns:
            Batch indexing results
        """
        success_count = 0
        error_count = 0
        errors = []
        
        for model_info in models:
            model_id = model_info.get('id') or Path(model_info['filepath']).stem
            filepath = model_info['filepath']
            cat = model_info.get('category', category)
            
            result = self.index_model(model_id, filepath, cat)
            
            if result.get('success'):
                success_count += 1
            else:
                error_count += 1
                errors.append({
                    'model_id': model_id,
                    'error': result.get('error')
                })
        
        return {
            'success': True,
            'indexed': success_count,
            'errors': error_count,
            'error_details': errors
        }
    
    def evaluate_search(self,
                        query_vector: np.ndarray,
                        query_category: str,
                        metric: str = 'cosine',
                        k_values: List[int] = None) -> Dict[str, float]:
        """
        Evaluate search performance with standard IR metrics.
        
        Args:
            query_vector: Query feature vector
            query_category: Ground truth category
            metric: Distance metric
            k_values: K values for P@K computation
            
        Returns:
            Dictionary of metric values
        """
        if k_values is None:
            k_values = [5, 10, 20]
        
        max_k = max(k_values)
        results = self.search(
            query_vector=query_vector,
            metric=metric,
            k=max_k
        )
        
        metrics = {}
        
        for k in k_values:
            top_k = results[:k]
            relevant = sum(1 for r in top_k if r['category'] == query_category)
            metrics[f'precision_at_{k}'] = relevant / k if k > 0 else 0
        
        # Compute Average Precision
        relevant_count = 0
        precision_sum = 0
        for i, r in enumerate(results):
            if r['category'] == query_category:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        total_relevant = len(self.index.get_models_by_category(query_category))
        metrics['average_precision'] = precision_sum / total_relevant if total_relevant > 0 else 0
        
        # Compute NDCG@K
        for k in k_values:
            dcg = 0
            idcg = 0
            for i in range(min(k, len(results))):
                rel = 1 if results[i]['category'] == query_category else 0
                dcg += rel / np.log2(i + 2)
            
            for i in range(min(k, total_relevant)):
                idcg += 1 / np.log2(i + 2)
            
            metrics[f'ndcg_at_{k}'] = dcg / idcg if idcg > 0 else 0
        
        return metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.index.get_stats()


# Singleton instances
_index_path = Path(__file__).parent / 'model3d_index.json'
model3d_index = Model3DIndex(str(_index_path))
similarity3d_service = Similarity3DService(model3d_index)


def get_similarity_service() -> Similarity3DService:
    """Get the singleton similarity service."""
    return similarity3d_service
