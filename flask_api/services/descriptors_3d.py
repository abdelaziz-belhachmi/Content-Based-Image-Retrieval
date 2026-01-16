"""
3D Shape Descriptors based on Local Features

This module implements local feature-based descriptors for 3D shape retrieval:
- Spin Images (Johnson & Hebert, 1999)
- 3D Shape Contexts (Körtgen et al., 2003)
- Shape Index Histogram (Zaharia & Prêteux, 2001)
- Point Feature Histograms (Rusu et al., 2009)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
import logging

from .mesh_loader import Mesh3D

logger = logging.getLogger(__name__)


class SpinImageDescriptor:
    """
    Spin Images descriptor for 3D shape retrieval.
    
    Spin images are 2D histograms that encode the local surface geometry
    around each point based on cylindrical coordinates.
    
    Reference:
        Johnson, A.E. and Hebert, M. (1999). Using spin images for efficient
        object recognition in cluttered 3D scenes.
    """
    
    def __init__(self, 
                 image_width: int = 16,
                 image_height: int = 16,
                 support_radius: float = 0.5,
                 n_sample_points: int = 1000):
        """
        Initialize the Spin Image descriptor.
        
        Args:
            image_width: Width of the spin image (alpha bins)
            image_height: Height of the spin image (beta bins)
            support_radius: Radius of support region (relative to normalized mesh)
            n_sample_points: Number of points to sample for descriptor computation
        """
        self.image_width = image_width
        self.image_height = image_height
        self.support_radius = support_radius
        self.n_sample_points = n_sample_points
        self.bin_size = support_radius / image_width
        
    def compute_single_spin_image(self, 
                                   point: np.ndarray, 
                                   normal: np.ndarray,
                                   all_points: np.ndarray) -> np.ndarray:
        """
        Compute spin image for a single oriented point.
        
        Args:
            point: 3D point position
            normal: Normal vector at the point
            all_points: All surface points
            
        Returns:
            2D spin image array
        """
        spin_image = np.zeros((self.image_height, self.image_width))
        
        for other_point in all_points:
            # Vector from point to other_point
            diff = other_point - point
            
            # Beta: signed distance along normal
            beta = np.dot(diff, normal)
            
            # Alpha: perpendicular distance to normal axis
            alpha = np.sqrt(np.dot(diff, diff) - beta * beta)
            
            # Check if within support region
            if alpha < self.support_radius and abs(beta) < self.support_radius:
                # Compute bin indices
                alpha_bin = int(alpha / self.bin_size)
                beta_bin = int((beta + self.support_radius) / (2 * self.bin_size))
                
                # Clamp to valid range
                alpha_bin = min(alpha_bin, self.image_width - 1)
                beta_bin = min(max(beta_bin, 0), self.image_height - 1)
                
                spin_image[beta_bin, alpha_bin] += 1
        
        # Normalize
        total = np.sum(spin_image)
        if total > 0:
            spin_image = spin_image / total
            
        return spin_image
    
    def compute(self, mesh: Mesh3D) -> np.ndarray:
        """
        Compute the global Spin Image descriptor for a mesh.
        
        Args:
            mesh: Normalized 3D mesh
            
        Returns:
            1D feature vector (flattened aggregated spin images)
        """
        # Sample points on the surface
        n_samples = min(self.n_sample_points, len(mesh.vertices))
        points, normals = mesh.sample_points(n_samples, method='uniform')
        
        # Compute spin images for a subset of points (for efficiency)
        n_reference = min(100, len(points))
        ref_indices = np.random.choice(len(points), n_reference, replace=False)
        
        spin_images = []
        for idx in ref_indices:
            si = self.compute_single_spin_image(points[idx], normals[idx], points)
            spin_images.append(si.flatten())
        
        spin_images = np.array(spin_images)
        
        # Aggregate: compute mean spin image
        mean_spin_image = np.mean(spin_images, axis=0)
        
        # Also add histogram of spin image values for discrimination
        hist, _ = np.histogram(spin_images.flatten(), bins=32, range=(0, 1))
        hist = hist / np.sum(hist)
        
        # Concatenate features
        descriptor = np.concatenate([mean_spin_image, hist])
        
        return descriptor
    
    @property
    def descriptor_size(self) -> int:
        """Return the size of the descriptor vector."""
        return self.image_width * self.image_height + 32


class ShapeContext3DDescriptor:
    """
    3D Shape Contexts descriptor for 3D shape retrieval.
    
    Shape contexts encode the relative distribution of surface points
    around each reference point using a spherical histogram.
    
    Reference:
        Körtgen, M. et al. (2003). 3D shape matching with 3D shape contexts.
    """
    
    def __init__(self,
                 n_radial_bins: int = 5,
                 n_polar_bins: int = 6,
                 n_azimuth_bins: int = 12,
                 max_radius: float = 1.0,
                 n_sample_points: int = 1000,
                 log_scale: bool = True):
        """
        Initialize the 3D Shape Context descriptor.
        
        Args:
            n_radial_bins: Number of radial (distance) bins
            n_polar_bins: Number of polar angle bins (0 to pi)
            n_azimuth_bins: Number of azimuthal angle bins (0 to 2*pi)
            max_radius: Maximum radius for context computation
            n_sample_points: Number of points to sample
            log_scale: Whether to use logarithmic radial binning
        """
        self.n_radial_bins = n_radial_bins
        self.n_polar_bins = n_polar_bins
        self.n_azimuth_bins = n_azimuth_bins
        self.max_radius = max_radius
        self.n_sample_points = n_sample_points
        self.log_scale = log_scale
        
        # Pre-compute radial bin edges
        if log_scale:
            self.radial_edges = np.logspace(np.log10(0.01), np.log10(max_radius), n_radial_bins + 1)
        else:
            self.radial_edges = np.linspace(0, max_radius, n_radial_bins + 1)
    
    def compute_single_context(self, 
                                point: np.ndarray,
                                normal: np.ndarray,
                                all_points: np.ndarray) -> np.ndarray:
        """
        Compute shape context for a single point.
        
        Args:
            point: Reference point position
            normal: Normal at reference point (defines local z-axis)
            all_points: All surface points
            
        Returns:
            3D histogram (radial x polar x azimuth)
        """
        context = np.zeros((self.n_radial_bins, self.n_polar_bins, self.n_azimuth_bins))
        
        # Build local coordinate system
        z_axis = normal / (np.linalg.norm(normal) + 1e-10)
        
        # Create orthogonal x and y axes
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(z_axis, np.array([1, 0, 0]))
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
        y_axis = np.cross(z_axis, x_axis)
        
        for other_point in all_points:
            diff = other_point - point
            dist = np.linalg.norm(diff)
            
            if dist < 1e-10 or dist > self.max_radius:
                continue
            
            # Transform to local coordinates
            local_x = np.dot(diff, x_axis)
            local_y = np.dot(diff, y_axis)
            local_z = np.dot(diff, z_axis)
            
            # Compute spherical coordinates
            r = dist
            theta = np.arccos(np.clip(local_z / r, -1, 1))  # Polar angle [0, pi]
            phi = np.arctan2(local_y, local_x) + np.pi  # Azimuth [0, 2*pi]
            
            # Find bins
            if self.log_scale:
                r_bin = np.searchsorted(self.radial_edges[1:], r)
            else:
                r_bin = int(r / self.max_radius * self.n_radial_bins)
            
            theta_bin = int(theta / np.pi * self.n_polar_bins)
            phi_bin = int(phi / (2 * np.pi) * self.n_azimuth_bins)
            
            # Clamp to valid range
            r_bin = min(r_bin, self.n_radial_bins - 1)
            theta_bin = min(theta_bin, self.n_polar_bins - 1)
            phi_bin = min(phi_bin, self.n_azimuth_bins - 1)
            
            context[r_bin, theta_bin, phi_bin] += 1
        
        # Normalize
        total = np.sum(context)
        if total > 0:
            context = context / total
            
        return context
    
    def compute(self, mesh: Mesh3D) -> np.ndarray:
        """
        Compute the global Shape Context descriptor for a mesh.
        
        Args:
            mesh: Normalized 3D mesh
            
        Returns:
            1D feature vector
        """
        # Sample points
        n_samples = min(self.n_sample_points, len(mesh.vertices))
        points, normals = mesh.sample_points(n_samples, method='uniform')
        
        # Compute contexts for reference points
        n_reference = min(50, len(points))
        ref_indices = np.random.choice(len(points), n_reference, replace=False)
        
        contexts = []
        for idx in ref_indices:
            ctx = self.compute_single_context(points[idx], normals[idx], points)
            contexts.append(ctx.flatten())
        
        contexts = np.array(contexts)
        
        # Aggregate: mean and variance
        mean_context = np.mean(contexts, axis=0)
        var_context = np.var(contexts, axis=0)
        
        # Combine
        descriptor = np.concatenate([mean_context, var_context])
        
        return descriptor
    
    @property
    def descriptor_size(self) -> int:
        """Return the size of the descriptor vector."""
        single_size = self.n_radial_bins * self.n_polar_bins * self.n_azimuth_bins
        return 2 * single_size  # mean + variance


class ShapeIndexHistogram:
    """
    Shape Index Histogram (3D Shape Spectrum Descriptor).
    
    Computes a histogram of shape index values over the mesh surface.
    The shape index characterizes local surface curvature.
    
    Reference:
        Zaharia, T. and Prêteux, F. (2001). 3D shape spectrum descriptor.
    """
    
    def __init__(self, n_bins: int = 64, n_sample_points: int = 1000):
        """
        Initialize the Shape Index Histogram descriptor.
        
        Args:
            n_bins: Number of histogram bins
            n_sample_points: Number of points for curvature estimation
        """
        self.n_bins = n_bins
        self.n_sample_points = n_sample_points
    
    def compute_curvatures(self, mesh: Mesh3D, k_neighbors: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute principal curvatures at each vertex using local quadric fitting.
        
        Args:
            mesh: 3D mesh
            k_neighbors: Number of neighbors for local fitting
            
        Returns:
            Tuple of (kappa1, kappa2) principal curvature arrays
        """
        n_vertices = len(mesh.vertices)
        kappa1 = np.zeros(n_vertices)
        kappa2 = np.zeros(n_vertices)
        
        # Build KD-tree for neighbor queries
        tree = cKDTree(mesh.vertices)
        
        for i in range(n_vertices):
            # Find k nearest neighbors
            distances, indices = tree.query(mesh.vertices[i], k=k_neighbors)
            
            if len(indices) < 6:  # Need enough points for quadric fitting
                continue
            
            # Get neighbor points in local coordinate system
            neighbors = mesh.vertices[indices]
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            
            # Local coordinate system based on normal
            normal = mesh.normals[i]
            if np.linalg.norm(normal) < 1e-10:
                continue
            
            normal = normal / np.linalg.norm(normal)
            
            # Create orthogonal basis
            if abs(normal[2]) < 0.9:
                u = np.cross(normal, np.array([0, 0, 1]))
            else:
                u = np.cross(normal, np.array([1, 0, 0]))
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)
            
            # Project to local coordinates
            local_x = np.dot(centered, u)
            local_y = np.dot(centered, v)
            local_z = np.dot(centered, normal)
            
            # Fit quadric: z = a*x^2 + b*x*y + c*y^2
            # Using least squares
            try:
                A = np.column_stack([local_x**2, local_x*local_y, local_y**2])
                coeffs, _, _, _ = np.linalg.lstsq(A, local_z, rcond=None)
                a, b, c = coeffs
                
                # Compute principal curvatures from quadric coefficients
                # H = trace of shape operator / 2
                # K = determinant of shape operator
                H = a + c
                K = 4 * a * c - b**2
                
                discriminant = H**2 - K
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    kappa1[i] = H + sqrt_disc
                    kappa2[i] = H - sqrt_disc
                else:
                    kappa1[i] = H
                    kappa2[i] = H
                    
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        return kappa1, kappa2
    
    def compute_shape_index(self, kappa1: np.ndarray, kappa2: np.ndarray) -> np.ndarray:
        """
        Compute shape index from principal curvatures.
        
        S = (2/pi) * arctan((k1 + k2) / (k1 - k2))
        
        Returns values in [-1, 1]
        """
        diff = kappa1 - kappa2
        sum_k = kappa1 + kappa2
        
        # Avoid division by zero
        shape_index = np.zeros_like(kappa1)
        valid = np.abs(diff) > 1e-10
        
        shape_index[valid] = (2 / np.pi) * np.arctan(sum_k[valid] / diff[valid])
        
        return shape_index
    
    def compute(self, mesh: Mesh3D) -> np.ndarray:
        """
        Compute the Shape Index Histogram descriptor.
        
        Args:
            mesh: 3D mesh
            
        Returns:
            1D histogram feature vector
        """
        # Compute principal curvatures
        kappa1, kappa2 = self.compute_curvatures(mesh)
        
        # Compute shape index
        shape_index = self.compute_shape_index(kappa1, kappa2)
        
        # Build histogram
        histogram, _ = np.histogram(shape_index, bins=self.n_bins, range=(-1, 1))
        
        # Normalize
        total = np.sum(histogram)
        if total > 0:
            histogram = histogram.astype(np.float64) / total
        
        # Apply smoothing
        histogram = gaussian_filter1d(histogram, sigma=1)
        
        return histogram
    
    @property
    def descriptor_size(self) -> int:
        """Return the size of the descriptor vector."""
        return self.n_bins


class PointFeatureHistogram:
    """
    Simplified Point Feature Histogram (PFH) descriptor.
    
    Encodes geometric relationships between pairs of points
    using angles derived from surface normals.
    
    Reference:
        Rusu, R.B. et al. (2009). Fast Point Feature Histograms (FPFH)
        for 3D registration.
    """
    
    def __init__(self, 
                 n_bins: int = 11,
                 search_radius: float = 0.2,
                 n_sample_points: int = 500):
        """
        Initialize the PFH descriptor.
        
        Args:
            n_bins: Number of bins per feature dimension
            search_radius: Radius for neighbor search
            n_sample_points: Number of points to sample
        """
        self.n_bins = n_bins
        self.search_radius = search_radius
        self.n_sample_points = n_sample_points
    
    def compute_pair_features(self, 
                               p1: np.ndarray, n1: np.ndarray,
                               p2: np.ndarray, n2: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute the 4 features for a pair of oriented points.
        
        Returns (alpha, phi, theta, d)
        """
        d = np.linalg.norm(p2 - p1)
        
        if d < 1e-10:
            return 0, 0, 0, 0
        
        # Define local coordinate frame at p1
        u = n1
        v = np.cross(u, (p2 - p1) / d)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return 0, 0, 0, d
        v = v / v_norm
        w = np.cross(u, v)
        
        # Compute angles
        alpha = np.dot(v, n2)
        phi = np.dot(u, (p2 - p1) / d)
        theta = np.arctan2(np.dot(w, n2), np.dot(u, n2))
        
        return alpha, phi, theta, d
    
    def compute(self, mesh: Mesh3D) -> np.ndarray:
        """
        Compute the PFH descriptor for a mesh.
        
        Args:
            mesh: Normalized 3D mesh
            
        Returns:
            1D feature vector
        """
        # Sample points
        n_samples = min(self.n_sample_points, len(mesh.vertices))
        points, normals = mesh.sample_points(n_samples, method='uniform')
        
        # Build KD-tree
        tree = cKDTree(points)
        
        # Collect all pair features
        all_alphas = []
        all_phis = []
        all_thetas = []
        
        for i in range(len(points)):
            # Find neighbors within radius
            neighbor_indices = tree.query_ball_point(points[i], self.search_radius)
            
            for j in neighbor_indices:
                if j <= i:
                    continue
                    
                alpha, phi, theta, d = self.compute_pair_features(
                    points[i], normals[i], points[j], normals[j]
                )
                
                all_alphas.append(alpha)
                all_phis.append(phi)
                all_thetas.append(theta)
        
        # Build histograms for each feature
        if len(all_alphas) == 0:
            return np.zeros(3 * self.n_bins)
        
        hist_alpha, _ = np.histogram(all_alphas, bins=self.n_bins, range=(-1, 1))
        hist_phi, _ = np.histogram(all_phis, bins=self.n_bins, range=(-1, 1))
        hist_theta, _ = np.histogram(all_thetas, bins=self.n_bins, range=(-np.pi, np.pi))
        
        # Normalize
        for hist in [hist_alpha, hist_phi, hist_theta]:
            total = np.sum(hist)
            if total > 0:
                hist[:] = hist / total
        
        descriptor = np.concatenate([hist_alpha, hist_phi, hist_theta]).astype(np.float64)
        
        return descriptor
    
    @property
    def descriptor_size(self) -> int:
        """Return the size of the descriptor vector."""
        return 3 * self.n_bins


class CombinedLocalDescriptor:
    """
    Combined descriptor using multiple local feature methods.
    """
    
    def __init__(self):
        """Initialize all descriptor components."""
        self.spin_image = SpinImageDescriptor()
        self.shape_context = ShapeContext3DDescriptor()
        self.shape_index = ShapeIndexHistogram()
        self.pfh = PointFeatureHistogram()
        
        self.weights = {
            'spin_image': 1.0,
            'shape_context': 1.0,
            'shape_index': 1.0,
            'pfh': 1.0
        }
    
    def compute(self, mesh: Mesh3D, descriptors: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Compute multiple descriptors for a mesh.
        
        Args:
            mesh: 3D mesh
            descriptors: List of descriptors to compute. If None, compute all.
            
        Returns:
            Dictionary of descriptor name -> feature vector
        """
        if descriptors is None:
            descriptors = ['spin_image', 'shape_context', 'shape_index', 'pfh']
        
        result = {}
        
        if 'spin_image' in descriptors:
            try:
                result['spin_image'] = self.spin_image.compute(mesh)
            except Exception as e:
                logger.warning(f"Failed to compute spin image: {e}")
                result['spin_image'] = np.zeros(self.spin_image.descriptor_size)
        
        if 'shape_context' in descriptors:
            try:
                result['shape_context'] = self.shape_context.compute(mesh)
            except Exception as e:
                logger.warning(f"Failed to compute shape context: {e}")
                result['shape_context'] = np.zeros(self.shape_context.descriptor_size)
        
        if 'shape_index' in descriptors:
            try:
                result['shape_index'] = self.shape_index.compute(mesh)
            except Exception as e:
                logger.warning(f"Failed to compute shape index: {e}")
                result['shape_index'] = np.zeros(self.shape_index.descriptor_size)
        
        if 'pfh' in descriptors:
            try:
                result['pfh'] = self.pfh.compute(mesh)
            except Exception as e:
                logger.warning(f"Failed to compute PFH: {e}")
                result['pfh'] = np.zeros(self.pfh.descriptor_size)
        
        return result
    
    def compute_combined_vector(self, mesh: Mesh3D) -> np.ndarray:
        """
        Compute a single combined feature vector.
        
        Args:
            mesh: 3D mesh
            
        Returns:
            Combined feature vector
        """
        descriptors = self.compute(mesh)
        
        # Concatenate with weighting
        vectors = []
        for name, weight in self.weights.items():
            if name in descriptors:
                vectors.append(weight * descriptors[name])
        
        return np.concatenate(vectors)
    
    @property
    def combined_size(self) -> int:
        """Return the size of the combined descriptor."""
        return (self.spin_image.descriptor_size + 
                self.shape_context.descriptor_size + 
                self.shape_index.descriptor_size + 
                self.pfh.descriptor_size)


# Singleton instance
descriptor_extractor = CombinedLocalDescriptor()


def extract_descriptors(mesh: Mesh3D, descriptor_types: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Extract local feature descriptors from a mesh.
    
    Args:
        mesh: 3D mesh
        descriptor_types: List of descriptor types to compute
        
    Returns:
        Dictionary of descriptor arrays
    """
    return descriptor_extractor.compute(mesh, descriptor_types)


def extract_combined_descriptor(mesh: Mesh3D) -> np.ndarray:
    """
    Extract a combined descriptor vector from a mesh.
    
    Args:
        mesh: 3D mesh
        
    Returns:
        Combined feature vector
    """
    return descriptor_extractor.compute_combined_vector(mesh)
