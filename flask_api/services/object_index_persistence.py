"""
Object Index Persistence Module
Saves and loads the object search index to/from disk.
Uses JSON for better compatibility and debugging.
"""

import os
import json

INDEX_PATH = os.path.join(os.path.dirname(__file__), 'object_index.json')


def save_object_index(obj):
    """
    Save object index to disk as JSON.
    
    Args:
        obj: Dictionary containing index data
    """
    try:
        with open(INDEX_PATH, 'w') as f:
            json.dump(obj, f, indent=2)
        print(f"[PERSISTENCE] Saved index to {INDEX_PATH}")
    except Exception as e:
        print(f"[PERSISTENCE ERROR] Failed to save index: {e}")


def load_object_index():
    """
    Load object index from disk.
    
    Returns:
        Dictionary with index data, or None if file doesn't exist
    """
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, 'r') as f:
                data = json.load(f)
            print(f"[PERSISTENCE] Loaded index from {INDEX_PATH}")
            return data
        except Exception as e:
            print(f"[PERSISTENCE ERROR] Failed to load index: {e}")
            return None
    else:
        print(f"[PERSISTENCE] No index file found at {INDEX_PATH}")
        return None