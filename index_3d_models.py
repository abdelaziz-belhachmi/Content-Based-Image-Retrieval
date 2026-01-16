#!/usr/bin/env python3
"""
3D Model Batch Indexer

Script to index 3D models from the downloaded benchmark dataset.
This can be run standalone or integrated with the Flask API.

Usage:
    python index_3d_models.py [--data-dir DATA_DIR] [--limit N] [--metric METRIC]
    
Examples:
    python index_3d_models.py
    python index_3d_models.py --data-dir Data_3D/models --limit 100
    python index_3d_models.py --evaluate --metric cosine
"""

import os
import sys
import argparse
from pathlib import Path
import time
import json

# Add flask_api to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
FLASK_API_DIR = SCRIPT_DIR / 'flask_api'
sys.path.insert(0, str(FLASK_API_DIR))

# Default paths
DEFAULT_DATA_DIR = SCRIPT_DIR / 'Data_3D' / 'models'
DEFAULT_INDEX_PATH = FLASK_API_DIR / 'services' / 'model3d_index.json'


def get_category_from_filename(filename: str) -> str:
    """
    Extract category from filename.
    
    The 3D Pottery dataset has model names that can be parsed
    to extract the category/class.
    
    Args:
        filename: Model filename (without path)
        
    Returns:
        Category string
    """
    # Try to extract category from filename pattern
    # Common patterns: "Amphora_001.obj", "amphora-1.obj", etc.
    name = Path(filename).stem.lower()
    
    # Known categories in 3D Pottery dataset
    categories = [
        'amphora', 'alabastron', 'hydria', 'kantharos', 'krater',
        'kylix', 'lekythos', 'oinochoe', 'psykter', 'pyxis',
        'skyphos', 'stamnos', 'bowl', 'jar', 'bottle', 'vase'
    ]
    
    for cat in categories:
        if cat in name:
            return cat.capitalize()
    
    # Default
    return 'Unknown'


def index_models(data_dir: Path, limit: int = None, verbose: bool = True) -> dict:
    """
    Index 3D models from a directory.
    
    Args:
        data_dir: Directory containing OBJ files
        limit: Maximum number of models to index (None for all)
        verbose: Print progress
        
    Returns:
        Dictionary with indexing results
    """
    from services.similarity_3d import get_similarity_service
    
    service = get_similarity_service()
    
    # Find all OBJ files
    obj_files = list(data_dir.glob('*.obj'))
    if not obj_files:
        obj_files = list(data_dir.rglob('*.obj'))
    
    if limit:
        obj_files = obj_files[:limit]
    
    if verbose:
        print(f"\nFound {len(obj_files)} OBJ files to index")
        print("=" * 50)
    
    results = {
        'indexed': 0,
        'errors': 0,
        'error_files': [],
        'categories': {}
    }
    
    start_time = time.time()
    
    for i, obj_file in enumerate(obj_files):
        model_id = obj_file.stem
        category = get_category_from_filename(obj_file.name)
        
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"\r  Indexing: {i + 1}/{len(obj_files)} ({rate:.1f} models/sec)", end='')
        
        try:
            result = service.index_model(
                model_id=model_id,
                filepath=str(obj_file),
                category=category,
                metadata={'source': 'batch_index'}
            )
            
            if result.get('success'):
                results['indexed'] += 1
                results['categories'][category] = results['categories'].get(category, 0) + 1
            else:
                results['errors'] += 1
                results['error_files'].append({
                    'file': str(obj_file),
                    'error': result.get('error')
                })
                
        except Exception as e:
            results['errors'] += 1
            results['error_files'].append({
                'file': str(obj_file),
                'error': str(e)
            })
    
    elapsed = time.time() - start_time
    results['time_seconds'] = elapsed
    results['rate'] = results['indexed'] / elapsed if elapsed > 0 else 0
    
    if verbose:
        print(f"\n\nIndexing complete!")
        print(f"  Indexed: {results['indexed']} models")
        print(f"  Errors: {results['errors']}")
        print(f"  Time: {elapsed:.1f} seconds ({results['rate']:.1f} models/sec)")
        print(f"\nCategories:")
        for cat, count in sorted(results['categories'].items()):
            print(f"    {cat}: {count}")
    
    return results


def evaluate_system(metric: str = 'cosine', k_values: list = None, verbose: bool = True) -> dict:
    """
    Evaluate the 3D retrieval system.
    
    Args:
        metric: Distance metric to use
        k_values: List of K values for P@K and NDCG@K
        verbose: Print progress
        
    Returns:
        Evaluation metrics
    """
    from services.similarity_3d import get_similarity_service
    
    if k_values is None:
        k_values = [5, 10, 20]
    
    service = get_similarity_service()
    
    if verbose:
        print(f"\nEvaluating system with metric: {metric}")
        print(f"K values: {k_values}")
        print("=" * 50)
    
    results = service.evaluate(metric=metric, k_values=k_values)
    
    if verbose:
        print(f"\nResults:")
        print(f"  mAP: {results.get('mAP', 0):.4f}")
        print(f"  Queries: {results.get('num_queries', 0)}")
        print(f"\nPrecision@K:")
        for k, p in results.get('precision_at_k', {}).items():
            print(f"    P@{k}: {p:.4f}")
        print(f"\nNDCG@K:")
        for k, n in results.get('ndcg_at_k', {}).items():
            print(f"    NDCG@{k}: {n:.4f}")
    
    return results


def get_stats(verbose: bool = True) -> dict:
    """Get current index statistics."""
    from services.similarity_3d import get_similarity_service
    
    service = get_similarity_service()
    stats = service.get_stats()
    
    if verbose:
        print(f"\n3D Index Statistics:")
        print("=" * 50)
        print(f"  Total models: {stats['total_models']}")
        print(f"  Index path: {stats['index_path']}")
        print(f"\nCategories:")
        for cat, count in sorted(stats.get('categories', {}).items()):
            print(f"    {cat}: {count}")
    
    return stats


def clear_index(verbose: bool = True) -> None:
    """Clear the 3D model index."""
    from services.similarity_3d import get_similarity_service
    
    service = get_similarity_service()
    service.index.clear()
    
    if verbose:
        print("3D index cleared.")


def main():
    parser = argparse.ArgumentParser(
        description='3D Model Batch Indexer for CBIR System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Index all models:
    python index_3d_models.py
    
  Index with limit:
    python index_3d_models.py --limit 100
    
  Index from custom directory:
    python index_3d_models.py --data-dir /path/to/models
    
  Evaluate system:
    python index_3d_models.py --evaluate
    
  Show statistics:
    python index_3d_models.py --stats
    
  Clear index:
    python index_3d_models.py --clear
"""
    )
    
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='Directory containing OBJ files')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of models to index')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate system after indexing')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'manhattan', 'correlation'],
                        help='Distance metric for evaluation')
    parser.add_argument('--stats', action='store_true',
                        help='Show index statistics')
    parser.add_argument('--clear', action='store_true',
                        help='Clear the index')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    print("=" * 70)
    print("   3D Model Batch Indexer")
    print("   CBIR System - Content-Based 3D Model Retrieval")
    print("=" * 70)
    
    # Clear index if requested
    if args.clear:
        clear_index(verbose)
        return
    
    # Show stats if requested
    if args.stats:
        get_stats(verbose)
        return
    
    # Index models
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\nError: Directory not found: {data_dir}")
        print(f"\nRun 'python download_3d_benchmark.py' first to download the dataset.")
        sys.exit(1)
    
    results = index_models(data_dir, limit=args.limit, verbose=verbose)
    
    # Evaluate if requested
    if args.evaluate:
        evaluate_system(metric=args.metric, verbose=verbose)
    
    # Show final stats
    if verbose:
        print("\n")
        get_stats(verbose)


if __name__ == '__main__':
    main()
