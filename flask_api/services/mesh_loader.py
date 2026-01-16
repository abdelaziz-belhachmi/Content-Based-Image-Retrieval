"""
3D Model Loader Service

Handles loading and preprocessing of 3D models in OBJ format.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Mesh3D:
    """Represents a 3D mesh with vertices, faces, and normals."""
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: Optional[np.ndarray] = None):
        """
        Initialize a 3D mesh.
        
        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of triangle face indices
            normals: Nx3 array of vertex normals (optional, will be computed if not provided)
        """
        self.vertices = vertices.astype(np.float64)
        self.faces = faces.astype(np.int32)
        
        if normals is not None:
            self.normals = normals.astype(np.float64)
        else:
            self.normals = self._compute_vertex_normals()
        
        # Compute bounding box
        self.bbox_min = np.min(vertices, axis=0)
        self.bbox_max = np.max(vertices, axis=0)
        self.centroid = np.mean(vertices, axis=0)
        
    def _compute_vertex_normals(self) -> np.ndarray:
        """Compute vertex normals by averaging face normals."""
        normals = np.zeros_like(self.vertices)
        
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(face_normal)
            if norm > 1e-10:
                face_normal = face_normal / norm
            
            # Add to vertex normals
            for idx in face:
                normals[idx] += face_normal
        
        # Normalize vertex normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        normals = normals / norms
        
        return normals
    
    def normalize(self) -> 'Mesh3D':
        """Normalize the mesh to fit in a unit sphere centered at origin."""
        # Center at origin
        centered_vertices = self.vertices - self.centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(centered_vertices, axis=1))
        if max_dist > 1e-10:
            normalized_vertices = centered_vertices / max_dist
        else:
            normalized_vertices = centered_vertices
        
        return Mesh3D(normalized_vertices, self.faces.copy(), self.normals.copy())
    
    def sample_points(self, n_points: int, method: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points uniformly on the mesh surface.
        
        Args:
            n_points: Number of points to sample
            method: Sampling method ('uniform' or 'vertices')
            
        Returns:
            Tuple of (points, normals) arrays
        """
        # Ensure normals have same length as vertices
        if len(self.normals) != len(self.vertices):
            logger.warning(f"Normal count mismatch, recomputing normals")
            self.normals = self._compute_vertex_normals()
        
        if method == 'vertices':
            # Simply return vertices (or subsample)
            if len(self.vertices) <= n_points:
                return self.vertices.copy(), self.normals.copy()
            indices = np.random.choice(len(self.vertices), n_points, replace=False)
            return self.vertices[indices], self.normals[indices]
        
        # Uniform sampling on triangles
        # Compute face areas
        face_areas = np.zeros(len(self.faces))
        for i, face in enumerate(self.faces):
            # Validate face indices
            if np.any(face >= len(self.vertices)) or np.any(face < 0):
                continue
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_areas[i] = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        
        # Normalize to get probabilities
        total_area = np.sum(face_areas)
        if total_area < 1e-10:
            # Fallback to vertex sampling
            return self.sample_points(n_points, method='vertices')
        
        face_probs = face_areas / total_area
        
        # Sample faces according to area
        sampled_faces = np.random.choice(len(self.faces), n_points, p=face_probs)
        
        # Sample points uniformly within each triangle
        points = np.zeros((n_points, 3))
        normals = np.zeros((n_points, 3))
        
        for i, face_idx in enumerate(sampled_faces):
            face = self.faces[face_idx]
            
            # Validate face indices
            if np.any(face >= len(self.vertices)) or np.any(face < 0):
                # Use fallback: pick a random valid vertex
                valid_idx = np.random.randint(0, len(self.vertices))
                points[i] = self.vertices[valid_idx]
                normals[i] = self.normals[valid_idx]
                continue
            
            v0, v1, v2 = self.vertices[face]
            n0, n1, n2 = self.normals[face]
            
            # Random barycentric coordinates
            r1, r2 = np.random.random(2)
            sqrt_r1 = np.sqrt(r1)
            u = 1 - sqrt_r1
            v = r2 * sqrt_r1
            w = 1 - u - v
            
            points[i] = u * v0 + v * v1 + w * v2
            normals[i] = u * n0 + v * n1 + w * n2
            
            # Normalize
            norm = np.linalg.norm(normals[i])
            if norm > 1e-10:
                normals[i] /= norm
        
        return points, normals
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mesh statistics."""
        return {
            'n_vertices': len(self.vertices),
            'n_faces': len(self.faces),
            'bbox_min': self.bbox_min.tolist(),
            'bbox_max': self.bbox_max.tolist(),
            'centroid': self.centroid.tolist(),
            'dimensions': (self.bbox_max - self.bbox_min).tolist()
        }


class OBJLoader:
    """Loader for Wavefront OBJ files."""
    
    @staticmethod
    def load(filepath: str) -> Mesh3D:
        """
        Load a 3D mesh from an OBJ file.
        
        Args:
            filepath: Path to the OBJ file
            
        Returns:
            Mesh3D object
        """
        vertices = []
        normals = []
        faces = []
        face_normals = []
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"OBJ file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v' and len(parts) >= 4:
                    # Vertex position
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
                        
                elif parts[0] == 'vn' and len(parts) >= 4:
                    # Vertex normal
                    try:
                        normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
                        
                elif parts[0] == 'f' and len(parts) >= 4:
                    # Face (triangle or polygon)
                    face_vertices = []
                    face_norms = []
                    
                    for part in parts[1:]:
                        # Parse face vertex index (format: v, v/vt, v/vt/vn, v//vn)
                        indices = part.split('/')
                        try:
                            v_idx = int(indices[0])
                            # OBJ indices are 1-based, convert to 0-based
                            v_idx = v_idx - 1 if v_idx > 0 else len(vertices) + v_idx
                            face_vertices.append(v_idx)
                            
                            if len(indices) >= 3 and indices[2]:
                                n_idx = int(indices[2])
                                n_idx = n_idx - 1 if n_idx > 0 else len(normals) + n_idx
                                face_norms.append(n_idx)
                        except (ValueError, IndexError):
                            continue
                    
                    # Triangulate polygon faces
                    if len(face_vertices) >= 3:
                        for i in range(1, len(face_vertices) - 1):
                            faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
                            if len(face_norms) == len(face_vertices):
                                face_normals.append([face_norms[0], face_norms[i], face_norms[i + 1]])
        
        if not vertices:
            raise ValueError("No vertices found in OBJ file")
        
        if not faces:
            raise ValueError("No faces found in OBJ file")
        
        vertices_array = np.array(vertices, dtype=np.float64)
        faces_array = np.array(faces, dtype=np.int32)
        
        # Validate face indices
        max_idx = np.max(faces_array)
        if max_idx >= len(vertices_array):
            logger.warning(f"Face indices exceed vertex count, clamping")
            faces_array = np.clip(faces_array, 0, len(vertices_array) - 1)
        
        # Note: We always compute vertex normals instead of using OBJ normals
        # because OBJ normals are per-face-vertex and may not match vertex count
        normals_array = None
        
        return Mesh3D(vertices_array, faces_array, normals_array)
    
    @staticmethod
    def save(mesh: Mesh3D, filepath: str, include_normals: bool = True) -> None:
        """
        Save a 3D mesh to an OBJ file.
        
        Args:
            mesh: Mesh3D object to save
            filepath: Output file path
            include_normals: Whether to include vertex normals
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# OBJ file generated by 3D Retrieval System\n")
            f.write(f"# Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}\n\n")
            
            # Write vertices
            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write normals
            if include_normals and mesh.normals is not None:
                f.write("\n")
                for n in mesh.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
            # Write faces
            f.write("\n")
            for face in mesh.faces:
                if include_normals:
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# Singleton loader instance
obj_loader = OBJLoader()


def load_mesh(filepath: str) -> Mesh3D:
    """Convenience function to load a mesh from file."""
    return obj_loader.load(filepath)


def load_and_normalize(filepath: str) -> Mesh3D:
    """Load a mesh and normalize it."""
    mesh = obj_loader.load(filepath)
    return mesh.normalize()
