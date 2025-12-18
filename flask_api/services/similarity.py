"""
Similarity Search Service
Implements various distance metrics for image similarity.
"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.preprocessing import normalize


class SimilarityService:
    """
    Service for computing similarity between feature vectors.
    """
    
    SUPPORTED_METRICS = ['cosine', 'euclidean', 'manhattan', 'chi_square', 'intersection']
    
    def __init__(self):
        self.feature_index = {}  # In-memory index: {image_id: feature_vector}
        self.metadata = {}  # {image_id: {class_name, bbox, ...}}
    
    def add_to_index(self, image_id, feature_vector, metadata=None):
        """
        Add a feature vector to the index.
        
        Args:
            image_id: Unique identifier for the image/object
            feature_vector: numpy array of features
            metadata: Optional dictionary with additional info
        """
        self.feature_index[image_id] = np.array(feature_vector)
        if metadata:
            self.metadata[image_id] = metadata
    
    def remove_from_index(self, image_id):
        """Remove an entry from the index."""
        if image_id in self.feature_index:
            del self.feature_index[image_id]
        if image_id in self.metadata:
            del self.metadata[image_id]
    
    def clear_index(self):
        """Clear the entire index."""
        self.feature_index.clear()
        self.metadata.clear()
    
    def get_index_size(self):
        """Get number of items in index."""
        return len(self.feature_index)
    
    # =====================
    # DISTANCE METRICS
    # =====================
    
    def cosine_distance(self, vec1, vec2):
        """
        Cosine distance: 1 - cosine_similarity
        Range: [0, 2], 0 means identical
        """
        return cosine(vec1, vec2)
    
    def euclidean_distance(self, vec1, vec2):
        """
        Euclidean (L2) distance
        """
        return euclidean(vec1, vec2)
    
    def manhattan_distance(self, vec1, vec2):
        """
        Manhattan (L1) distance
        """
        return cityblock(vec1, vec2)
    
    def chi_square_distance(self, vec1, vec2):
        """
        Chi-square distance - good for histograms
        """
        eps = 1e-10
        return 0.5 * np.sum(((vec1 - vec2) ** 2) / (vec1 + vec2 + eps))
    
    def histogram_intersection(self, vec1, vec2):
        """
        Histogram intersection similarity (converted to distance)
        """
        intersection = np.minimum(vec1, vec2).sum()
        # Convert to distance (1 - normalized intersection)
        return 1 - (intersection / (min(vec1.sum(), vec2.sum()) + 1e-10))
    
    def compute_distance(self, vec1, vec2, metric='cosine'):
        """
        Compute distance between two vectors using specified metric.
        
        Args:
            vec1, vec2: Feature vectors
            metric: One of 'cosine', 'euclidean', 'manhattan', 'chi_square', 'intersection'
            
        Returns:
            Distance value (lower = more similar)
        """
        vec1 = np.array(vec1, dtype=np.float64)
        vec2 = np.array(vec2, dtype=np.float64)
        
        # Handle NaN/Inf
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
            raise ValueError(f"Unknown metric: {metric}. Supported: {self.SUPPORTED_METRICS}")
    
    # =====================
    # SEARCH FUNCTIONS
    # =====================
    
    def search(self, query_vector, top_k=10, metric='cosine', filter_class=None):
        """
        Search for similar items in the index.
        
        Args:
            query_vector: Query feature vector
            top_k: Number of results to return
            metric: Distance metric to use
            filter_class: Optional class name to filter results
            
        Returns:
            List of (image_id, distance, metadata) tuples, sorted by distance
        """
        if not self.feature_index:
            return []
        
        query_vector = np.array(query_vector, dtype=np.float64)
        query_vector = np.nan_to_num(query_vector)
        
        results = []
        
        for image_id, feature_vector in self.feature_index.items():
            # Apply class filter if specified
            if filter_class:
                meta = self.metadata.get(image_id, {})
                if meta.get('class_name') != filter_class:
                    continue
            
            distance = self.compute_distance(query_vector, feature_vector, metric)
            
            results.append({
                'image_id': image_id,
                'distance': float(distance),
                'similarity': float(1 / (1 + distance)),  # Convert to similarity score
                'metadata': self.metadata.get(image_id, {})
            })
        
        # Sort by distance (ascending)
        results.sort(key=lambda x: x['distance'])
        
        return results[:top_k]
    
    def search_batch(self, query_vectors, top_k=10, metric='cosine'):
        """
        Search for multiple query vectors.
        
        Args:
            query_vectors: List of query feature vectors
            top_k: Number of results per query
            metric: Distance metric
            
        Returns:
            List of search results for each query
        """
        return [self.search(qv, top_k, metric) for qv in query_vectors]
    
    def find_duplicates(self, threshold=0.1, metric='cosine'):
        """
        Find potential duplicate images in the index.
        
        Args:
            threshold: Maximum distance to consider as duplicate
            metric: Distance metric
            
        Returns:
            List of (image_id1, image_id2, distance) tuples
        """
        duplicates = []
        image_ids = list(self.feature_index.keys())
        
        for i in range(len(image_ids)):
            for j in range(i + 1, len(image_ids)):
                id1, id2 = image_ids[i], image_ids[j]
                distance = self.compute_distance(
                    self.feature_index[id1],
                    self.feature_index[id2],
                    metric
                )
                
                if distance < threshold:
                    duplicates.append({
                        'image_id_1': id1,
                        'image_id_2': id2,
                        'distance': float(distance)
                    })
        
        return duplicates
    
    # =====================
    # FEATURE NORMALIZATION
    # =====================
    
    @staticmethod
    def normalize_vector(vector, method='l2'):
        """
        Normalize a feature vector.
        
        Args:
            vector: Feature vector
            method: 'l2', 'l1', or 'minmax'
            
        Returns:
            Normalized vector
        """
        vector = np.array(vector, dtype=np.float64).reshape(1, -1)
        
        if method == 'l2':
            return normalize(vector, norm='l2').flatten()
        elif method == 'l1':
            return normalize(vector, norm='l1').flatten()
        elif method == 'minmax':
            min_val = vector.min()
            max_val = vector.max()
            if max_val - min_val > 0:
                return ((vector - min_val) / (max_val - min_val)).flatten()
            return vector.flatten()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    # =====================
    # PERSISTENCE
    # =====================
    
    def save_index(self, filepath):
        """Save index to file."""
        import json
        
        data = {
            'features': {k: v.tolist() for k, v in self.feature_index.items()},
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_index(self, filepath):
        """Load index from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.feature_index = {k: np.array(v) for k, v in data['features'].items()}
        self.metadata = data['metadata']


# Singleton instance
similarity_service = SimilarityService()
