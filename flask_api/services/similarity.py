"""
Similarity Search Service
Implements various distance metrics for image similarity.
Includes CBIR evaluation metrics: P@K, mAP@K, NDCG@K
"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Optional


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
    
    # =====================
    # CBIR EVALUATION METRICS
    # =====================
    
    def precision_at_k(self, retrieved_results: List[Dict], query_class: str, k: int = 10) -> float:
        """
        Precision@K: Proportion of relevant items in top-K results.
        
        P@K = (# of relevant items in top K) / K
        
        Args:
            retrieved_results: List of search results with metadata
            query_class: The class name of the query image
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if not retrieved_results or k <= 0:
            return 0.0
        
        top_k = retrieved_results[:k]
        relevant_count = sum(
            1 for result in top_k
            if result.get('metadata', {}).get('class_name') == query_class
        )
        
        return relevant_count / k
    
    def average_precision_at_k(self, retrieved_results: List[Dict], query_class: str, k: int = 50) -> float:
        """
        Average Precision@K: Mean of precision values at each relevant item.
        
        AP@K = (1/R) * Σ(P@i * rel(i)) for i=1 to K
        where R = number of relevant items, rel(i) = 1 if item i is relevant
        
        Args:
            retrieved_results: List of search results with metadata
            query_class: The class name of the query image
            k: Maximum number of results to consider
            
        Returns:
            AP@K score (0.0 to 1.0)
        """
        if not retrieved_results or k <= 0:
            return 0.0
        
        top_k = retrieved_results[:k]
        relevant_count = 0
        precision_sum = 0.0
        
        for i, result in enumerate(top_k, 1):
            if result.get('metadata', {}).get('class_name') == query_class:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / relevant_count
    
    def mean_average_precision_at_k(self, query_results_list: List[Dict], k: int = 50) -> float:
        """
        Mean Average Precision@K: Average of AP@K across multiple queries.
        
        mAP@K = (1/Q) * Σ AP@K(q) for all queries q
        
        Args:
            query_results_list: List of dicts, each containing:
                - 'results': search results for a query
                - 'query_class': class name of the query
            k: Maximum number of results to consider
            
        Returns:
            mAP@K score (0.0 to 1.0)
        """
        if not query_results_list:
            return 0.0
        
        ap_scores = []
        for query_data in query_results_list:
            ap = self.average_precision_at_k(
                query_data.get('results', []),
                query_data.get('query_class', ''),
                k
            )
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def dcg_at_k(self, retrieved_results: List[Dict], query_class: str, k: int = 10) -> float:
        """
        Discounted Cumulative Gain@K: Measures ranking quality with position discount.
        
        DCG@K = Σ(rel(i) / log2(i+1)) for i=1 to K
        rel(i) = 1 if result i is relevant, 0 otherwise
        
        Args:
            retrieved_results: List of search results with metadata
            query_class: The class name of the query image
            k: Number of top results to consider
            
        Returns:
            DCG@K score
        """
        if not retrieved_results or k <= 0:
            return 0.0
        
        top_k = retrieved_results[:k]
        dcg = 0.0
        
        for i, result in enumerate(top_k, 1):
            if result.get('metadata', {}).get('class_name') == query_class:
                dcg += 1.0 / np.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(self, retrieved_results: List[Dict], query_class: str, 
                  total_relevant: Optional[int] = None, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K: DCG normalized by ideal DCG.
        
        NDCG@K = DCG@K / IDCG@K
        where IDCG@K is the DCG of an ideal ranking (all relevant items first)
        
        Args:
            retrieved_results: List of search results with metadata
            query_class: The class name of the query image
            total_relevant: Total number of relevant items in the database.
                           If None, will count relevant items in results
            k: Number of top results to consider
            
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if not retrieved_results or k <= 0:
            return 0.0
        
        dcg = self.dcg_at_k(retrieved_results, query_class, k)
        
        if dcg == 0.0:
            return 0.0
        
        # Calculate ideal DCG (all relevant items first)
        if total_relevant is None:
            # Count relevant items in results
            total_relevant = sum(
                1 for result in retrieved_results
                if result.get('metadata', {}).get('class_name') == query_class
            )
        
        # IDCG: best possible DCG (all relevant items ranked first)
        num_relevant_in_k = min(total_relevant, k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant_in_k + 1))
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_search(self, query_vector, query_class: str, 
                        metric: str = 'cosine', 
                        top_k: int = 50,
                        total_relevant: Optional[int] = None) -> Dict[str, float]:
        """
        Perform a search and calculate all CBIR evaluation metrics.
        
        Args:
            query_vector: Feature vector of the query image
            query_class: Class name of the query image
            metric: Distance metric to use
            top_k: Maximum results for search
            total_relevant: Total relevant items in database (optional)
            
        Returns:
            Dict containing all evaluation metrics
        """
        results = self.search(query_vector, top_k=top_k, metric=metric)
        
        return {
            'precision_at_10': self.precision_at_k(results, query_class, k=10),
            'precision_at_5': self.precision_at_k(results, query_class, k=5),
            'ap_at_50': self.average_precision_at_k(results, query_class, k=50),
            'ndcg_at_10': self.ndcg_at_k(results, query_class, total_relevant, k=10),
            'ndcg_at_5': self.ndcg_at_k(results, query_class, total_relevant, k=5),
            'num_results': len(results),
            'query_class': query_class
        }
    
    def batch_evaluate(self, queries: List[Dict], metric: str = 'cosine', 
                       top_k: int = 50) -> Dict[str, float]:
        """
        Evaluate retrieval performance across multiple queries.
        
        Args:
            queries: List of dicts with 'vector' and 'class_name'
            metric: Distance metric
            top_k: Maximum results per query
            
        Returns:
            Aggregated metrics including mAP@50
        """
        all_results = []
        p_at_10_scores = []
        ndcg_at_10_scores = []
        
        for query in queries:
            results = self.search(query['vector'], top_k=top_k, metric=metric)
            query_class = query['class_name']
            
            all_results.append({
                'results': results,
                'query_class': query_class
            })
            
            p_at_10_scores.append(self.precision_at_k(results, query_class, k=10))
            ndcg_at_10_scores.append(self.ndcg_at_k(results, query_class, k=10))
        
        return {
            'mean_precision_at_10': float(np.mean(p_at_10_scores)) if p_at_10_scores else 0.0,
            'map_at_50': self.mean_average_precision_at_k(all_results, k=50),
            'mean_ndcg_at_10': float(np.mean(ndcg_at_10_scores)) if ndcg_at_10_scores else 0.0,
            'num_queries': len(queries)
        }


# Singleton instance
similarity_service = SimilarityService()
