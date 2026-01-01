"""
Enhanced Similarity Search Service - Object-Based Search
Implements similarity search at the OBJECT level, not image level.
NOW WITH AUTOMATIC PERSISTENCE!
"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.preprocessing import normalize
from collections import defaultdict
from .object_index_persistence import save_object_index, load_object_index


class ObjectBasedSimilarityService:
    """
    Service for computing similarity between OBJECTS, not entire images.
    Each indexed item represents a detected object with its features.
    """
    
    SUPPORTED_METRICS = ['cosine', 'euclidean', 'manhattan', 'chi_square', 'intersection']
    
    def __init__(self):
        # Index: {object_id: feature_vector}
        self.feature_index = {}
        
        # Metadata: {object_id: {image_id, class_name, bbox, ...}}
        self.metadata = {}
        
        # Reverse index: {image_id: [object_id1, object_id2, ...]}
        self.image_to_objects = defaultdict(list)
        
        # Class index: {class_name: [object_id1, object_id2, ...]}
        self.class_to_objects = defaultdict(list)
        
        # Load existing index on initialization
        self._load_index()
    
    # =====================
    # INDEX MANAGEMENT
    # =====================
    
    def add_object_to_index(self, object_id, feature_vector, metadata):
        """
        Add a detected object to the index.
        
        Args:
            object_id: Unique identifier for this object instance
            feature_vector: numpy array of features
            metadata: Dict with {image_id, class_name, bbox, confidence, ...}
        """
        self.feature_index[object_id] = np.array(feature_vector)
        self.metadata[object_id] = metadata
        
        # Update reverse indices
        image_id = metadata.get('image_id')
        class_name = metadata.get('class_name')
        
        if image_id:
            self.image_to_objects[image_id].append(object_id)
        if class_name:
            self.class_to_objects[class_name].append(object_id)
        
        self._save_index()
    
    def _load_index(self):
        """Load index from persistent storage on initialization."""
        data = load_object_index()
        if data:
            # Convert feature vectors back to numpy arrays
            self.feature_index = {k: np.array(v) for k, v in data.get('feature_index', {}).items()}
            self.metadata = data.get('metadata', {})
            self.image_to_objects = defaultdict(list, data.get('image_to_objects', {}))
            self.class_to_objects = defaultdict(list, data.get('class_to_objects', {}))
            print(f"[INDEX] Loaded {len(self.feature_index)} objects from disk")
        else:
            print("[INDEX] No existing index found, starting fresh")
    
    def remove_object_from_index(self, object_id):
        """Remove a specific object from the index."""
        if object_id not in self.feature_index:
            return
        
        metadata = self.metadata.get(object_id, {})
        image_id = metadata.get('image_id')
        class_name = metadata.get('class_name')
        
        # Clean up indices
        del self.feature_index[object_id]
        del self.metadata[object_id]
        
        if image_id and object_id in self.image_to_objects[image_id]:
            self.image_to_objects[image_id].remove(object_id)
        
        if class_name and object_id in self.class_to_objects[class_name]:
            self.class_to_objects[class_name].remove(object_id)
        
        self._save_index()
    
    def remove_image_from_index(self, image_id):
        """Remove all objects from a specific image."""
        if image_id not in self.image_to_objects:
            return
        
        object_ids = list(self.image_to_objects[image_id])
        for obj_id in object_ids:
            # Remove from feature index and metadata
            if obj_id in self.feature_index:
                del self.feature_index[obj_id]
            if obj_id in self.metadata:
                class_name = self.metadata[obj_id].get('class_name')
                if class_name and obj_id in self.class_to_objects[class_name]:
                    self.class_to_objects[class_name].remove(obj_id)
                del self.metadata[obj_id]
        
        del self.image_to_objects[image_id]
        self._save_index()
    
    def clear_index(self):
        """Clear the entire index."""
        self.feature_index.clear()
        self.metadata.clear()
        self.image_to_objects.clear()
        self.class_to_objects.clear()
        self._save_index()
    
    def get_index_size(self):
        """Get number of objects in index."""
        return len(self.feature_index)
    
    def _save_index(self):
        """Save all index data to disk."""
        # Convert numpy arrays to lists for JSON serialization
        save_object_index({
            'feature_index': {k: v.tolist() for k, v in self.feature_index.items()},
            'metadata': self.metadata,
            'image_to_objects': dict(self.image_to_objects),
            'class_to_objects': dict(self.class_to_objects)
        })
    
    # =====================
    # DISTANCE METRICS
    # =====================
    
    def cosine_distance(self, vec1, vec2):
        """Cosine distance: 1 - cosine_similarity"""
        return cosine(vec1, vec2)
    
    def euclidean_distance(self, vec1, vec2):
        """Euclidean (L2) distance"""
        return euclidean(vec1, vec2)
    
    def manhattan_distance(self, vec1, vec2):
        """Manhattan (L1) distance"""
        return cityblock(vec1, vec2)
    
    def chi_square_distance(self, vec1, vec2):
        """Chi-square distance - good for histograms"""
        eps = 1e-10
        return 0.5 * np.sum(((vec1 - vec2) ** 2) / (vec1 + vec2 + eps))
    
    def histogram_intersection(self, vec1, vec2):
        """Histogram intersection similarity (converted to distance)"""
        intersection = np.minimum(vec1, vec2).sum()
        return 1 - (intersection / (min(vec1.sum(), vec2.sum()) + 1e-10))
    
    def compute_distance(self, vec1, vec2, metric='cosine'):
        """Compute distance between two vectors using specified metric."""
        vec1 = np.array(vec1, dtype=np.float64)
        vec2 = np.array(vec2, dtype=np.float64)
        
        vec1 = np.nan_to_num(vec1)
        vec2 = np.nan_to_num(vec2)
        
        if metric == 'cosine':
            return self.cosine_distance(vec1, vec2)
        elif metric == 'euclidean':
            return self.euclidean_distance(vec1, vec2)
        elif metric == 'manhattan':
            return self.manhattan_distance(vec1, vec2)
        elif metric == 'chi_square':
            return self.chi_square_distance(vec1, vec2)
        elif metric == 'intersection':
            return self.histogram_intersection(vec1, vec2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # =====================
    # OBJECT-BASED SEARCH
    # =====================
    
    def search_by_object(self, query_vector, query_class, top_k=10, metric='cosine'):
        """
        Search for similar objects of the SAME CLASS.
        
        Args:
            query_vector: Feature vector of the query object
            query_class: Class name of the query object (e.g., 'bus', 'person')
            top_k: Number of results
            metric: Distance metric
            
        Returns:
            List of similar objects with their metadata
        """
        if query_class not in self.class_to_objects:
            return []
        
        query_vector = np.array(query_vector, dtype=np.float64)
        query_vector = np.nan_to_num(query_vector)
        
        results = []
        
        # Only search within the same class
        for object_id in self.class_to_objects[query_class]:
            feature_vector = self.feature_index[object_id]
            distance = self.compute_distance(query_vector, feature_vector, metric)
            
            results.append({
                'object_id': object_id,
                'distance': float(distance),
                'similarity': float(1 / (1 + distance)),
                'metadata': self.metadata[object_id]
            })
        
        # Sort by distance
        results.sort(key=lambda x: x['distance'])
        
        return results[:top_k]
    
    def search_by_image_objects(self, query_objects, top_k=10, metric='cosine', 
                                 aggregation='best_match', exclude_image_id=None):
        """
        Search for images containing similar objects to those in the query image.
        
        CRITICAL: Only returns images that contain AT LEAST ONE of the query object classes.
        
        Args:
            query_objects: List of {class_name, feature_vector} for each detected object
            top_k: Number of result images
            metric: Distance metric
            aggregation: How to aggregate scores:
                - 'best_match': Use best matching object per class
                - 'average': Average similarity across all objects
                - 'min_distance': Use minimum distance per image
                - 'any_match': Return images with ANY matching class
            exclude_image_id: Optional image ID to exclude from results
        
        Returns:
            List of images ranked by similarity, containing ONLY matching object classes
        """
        if not query_objects:
            return []
        
        # Extract query classes
        query_classes = {obj['class_name'] for obj in query_objects}
        
        print(f"[SEARCH] Looking for images with classes: {query_classes}")
        
        # Find all candidate images that contain AT LEAST ONE of the query classes
        candidate_images = set()
        for class_name in query_classes:
            if class_name in self.class_to_objects:
                for obj_id in self.class_to_objects[class_name]:
                    image_id = self.metadata[obj_id].get('image_id')
                    if image_id:
                        candidate_images.add(image_id)
        
        print(f"[SEARCH] Found {len(candidate_images)} candidate images")
        
        if not candidate_images:
            return []
        
        # Exclude query image if requested
        if exclude_image_id is not None:
            candidate_images = {img_id for img_id in candidate_images 
                              if str(img_id) != str(exclude_image_id)}
        
        if not candidate_images:
            return []
        
        # Score each candidate image
        image_scores = []
        
        for image_id in candidate_images:
            # Get all objects in this image
            image_object_ids = self.image_to_objects[image_id]
            
            # Check if image contains ANY of the query classes
            image_classes = {self.metadata[obj_id]['class_name'] for obj_id in image_object_ids}
            matching_classes = query_classes & image_classes
            
            if not matching_classes:
                continue  # Skip images without matching classes
            
            # Calculate similarity score based on aggregation method
            if aggregation == 'best_match':
                score = self._score_best_match(query_objects, image_object_ids, matching_classes, metric)
            elif aggregation == 'average':
                score = self._score_average(query_objects, image_object_ids, matching_classes, metric)
            elif aggregation == 'min_distance':
                score = self._score_min_distance(query_objects, image_object_ids, matching_classes, metric)
            elif aggregation == 'any_match':
                score = self._score_any_match(query_objects, image_object_ids, matching_classes, metric)
            else:
                score = self._score_best_match(query_objects, image_object_ids, matching_classes, metric)
            
            image_scores.append({
                'image_id': image_id,
                'score': score,
                'similarity': float(1 / (1 + score)),
                'matching_classes': list(matching_classes),
                'num_matching_objects': len([obj_id for obj_id in image_object_ids 
                                             if self.metadata[obj_id]['class_name'] in matching_classes])
            })
        
        # Sort by score (lower = better)
        image_scores.sort(key=lambda x: x['score'])
        
        print(f"[SEARCH] Returning {len(image_scores[:top_k])} results")
        
        return image_scores[:top_k]
    
    # =====================
    # SCORING METHODS
    # =====================
    
    def _score_best_match(self, query_objects, image_object_ids, matching_classes, metric):
        """Best matching object per class"""
        total_distance = 0
        num_matches = 0
        
        for query_obj in query_objects:
            if query_obj['class_name'] not in matching_classes:
                continue
            
            query_vector = query_obj['feature_vector']
            best_distance = float('inf')
            
            # Find best match for this query object in the image
            for obj_id in image_object_ids:
                obj_meta = self.metadata[obj_id]
                if obj_meta['class_name'] != query_obj['class_name']:
                    continue
                
                obj_vector = self.feature_index[obj_id]
                distance = self.compute_distance(query_vector, obj_vector, metric)
                best_distance = min(best_distance, distance)
            
            if best_distance != float('inf'):
                total_distance += best_distance
                num_matches += 1
        
        return total_distance / max(num_matches, 1)
    
    def _score_average(self, query_objects, image_object_ids, matching_classes, metric):
        """Average distance across all matching objects"""
        distances = []
        
        for query_obj in query_objects:
            if query_obj['class_name'] not in matching_classes:
                continue
            
            query_vector = query_obj['feature_vector']
            
            for obj_id in image_object_ids:
                obj_meta = self.metadata[obj_id]
                if obj_meta['class_name'] != query_obj['class_name']:
                    continue
                
                obj_vector = self.feature_index[obj_id]
                distance = self.compute_distance(query_vector, obj_vector, metric)
                distances.append(distance)
        
        return np.mean(distances) if distances else float('inf')
    
    def _score_min_distance(self, query_objects, image_object_ids, matching_classes, metric):
        """Minimum distance (most similar object pair)"""
        min_distance = float('inf')
        
        for query_obj in query_objects:
            if query_obj['class_name'] not in matching_classes:
                continue
            
            query_vector = query_obj['feature_vector']
            
            for obj_id in image_object_ids:
                obj_meta = self.metadata[obj_id]
                if obj_meta['class_name'] != query_obj['class_name']:
                    continue
                
                obj_vector = self.feature_index[obj_id]
                distance = self.compute_distance(query_vector, obj_vector, metric)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _score_any_match(self, query_objects, image_object_ids, matching_classes, metric):
        """Simply check if any class matches (binary matching)"""
        return 0.0 if matching_classes else float('inf')
    
    # =====================
    # STATISTICS
    # =====================
    
    def get_statistics(self):
        """Get index statistics"""
        return {
            'total_objects': len(self.feature_index),
            'total_images': len(self.image_to_objects),
            'classes': {
                class_name: len(object_ids) 
                for class_name, object_ids in self.class_to_objects.items()
            },
            'objects_per_image': {
                image_id: len(object_ids)
                for image_id, object_ids in self.image_to_objects.items()
            }
        }


# Singleton instance
object_similarity_service = ObjectBasedSimilarityService()