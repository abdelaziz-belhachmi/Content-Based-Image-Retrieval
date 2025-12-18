"""
Feature Extraction Service - Visual Descriptors
Implements color, texture, and shape descriptors.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.cluster import KMeans


class FeatureExtractor:
    """
    Extract visual features from images.
    Includes color, texture, and shape descriptors.
    """
    
    def __init__(self):
        # Gabor filter parameters
        self.gabor_scales = 4
        self.gabor_orientations = 6
        
        # LBP parameters
        self.lbp_radius = 3
        self.lbp_points = 24
        
        # Dominant colors
        self.n_dominant_colors = 5
    
    # =====================
    # COLOR DESCRIPTORS
    # =====================
    
    def extract_color_histogram_rgb(self, image, bins=32):
        """
        Extract RGB color histogram.
        
        Args:
            image: BGR image (numpy array)
            bins: Number of bins per channel
            
        Returns:
            Normalized histogram (bins * 3,)
        """
        hist_b = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
        hist_r = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
        
        hist = np.concatenate([hist_r, hist_g, hist_b])
        
        # Normalize
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist
    
    def extract_color_histogram_hsv(self, image, h_bins=30, s_bins=32, v_bins=32):
        """
        Extract HSV color histogram.
        
        Args:
            image: BGR image (numpy array)
            h_bins: Hue bins
            s_bins: Saturation bins
            v_bins: Value bins
            
        Returns:
            Normalized histogram
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).flatten()
        
        hist = np.concatenate([hist_h, hist_s, hist_v])
        
        # Normalize
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist
    
    def extract_dominant_colors(self, image, k=5):
        """
        Extract dominant colors using K-means clustering.
        
        Args:
            image: BGR image
            k: Number of dominant colors
            
        Returns:
            Dictionary with colors (RGB) and their percentages
        """
        # Reshape image to pixels
        pixels = image.reshape(-1, 3)
        
        # Convert to float
        pixels = np.float32(pixels)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (colors) and labels
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Calculate percentages
        label_counts = np.bincount(labels, minlength=k)
        percentages = label_counts / len(labels)
        
        # Sort by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        
        result = {
            'colors': [],
            'percentages': []
        }
        
        for idx in sorted_indices:
            # Convert BGR to RGB
            bgr = colors[idx]
            rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
            result['colors'].append(rgb)
            result['percentages'].append(float(percentages[idx]))
        
        return result
    
    def extract_color_moments(self, image):
        """
        Extract color moments (mean, std, skewness) for each channel.
        
        Returns:
            Array of 9 values (3 moments × 3 channels)
        """
        # Convert to float
        img_float = image.astype(np.float64)
        
        moments = []
        
        for channel in cv2.split(img_float):
            mean = np.mean(channel)
            std = np.std(channel)
            
            # Skewness
            if std > 0:
                skewness = np.mean(((channel - mean) / std) ** 3)
            else:
                skewness = 0
            
            moments.extend([mean / 255.0, std / 255.0, skewness])
        
        return np.array(moments)
    
    # =====================
    # TEXTURE DESCRIPTORS
    # =====================
    
    def extract_tamura_features(self, image):
        """
        Extract Tamura texture features.
        
        Returns:
            Dictionary with coarseness, contrast, directionality
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float64)
        
        # 1. Coarseness
        coarseness = self._tamura_coarseness(gray)
        
        # 2. Contrast
        contrast = self._tamura_contrast(gray)
        
        # 3. Directionality
        directionality = self._tamura_directionality(gray)
        
        return {
            'coarseness': float(coarseness),
            'contrast': float(contrast),
            'directionality': float(directionality)
        }
    
    def _tamura_coarseness(self, gray, kmax=5):
        """Calculate Tamura coarseness"""
        h, w = gray.shape
        
        # Calculate moving averages at different scales
        averages = {}
        for k in range(1, kmax + 1):
            size = 2 ** k
            kernel = np.ones((size, size)) / (size * size)
            averages[k] = cv2.filter2D(gray, -1, kernel)
        
        # Calculate best scale for each pixel
        best_k = np.ones((h, w))
        max_e = np.zeros((h, w))
        
        for k in range(1, kmax):
            size = 2 ** k
            
            # Horizontal difference
            e_h = np.abs(
                np.roll(averages[k], -size, axis=1) - 
                np.roll(averages[k], size, axis=1)
            )
            
            # Vertical difference
            e_v = np.abs(
                np.roll(averages[k], -size, axis=0) - 
                np.roll(averages[k], size, axis=0)
            )
            
            e = np.maximum(e_h, e_v)
            
            mask = e > max_e
            best_k[mask] = k
            max_e[mask] = e[mask]
        
        # Coarseness is average of 2^k
        coarseness = np.mean(2 ** best_k)
        
        return coarseness
    
    def _tamura_contrast(self, gray):
        """Calculate Tamura contrast"""
        # Contrast based on standard deviation and kurtosis
        mean = np.mean(gray)
        std = np.std(gray)
        
        if std == 0:
            return 0
        
        # Fourth moment (kurtosis)
        kurtosis = np.mean(((gray - mean) / std) ** 4)
        
        # Contrast formula
        contrast = std / (kurtosis ** 0.25 + 1e-6)
        
        return contrast / 255.0  # Normalize
    
    def _tamura_directionality(self, gray):
        """Calculate Tamura directionality"""
        # Gradient magnitude and direction
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        # Threshold for edge pixels
        threshold = np.mean(magnitude)
        mask = magnitude > threshold
        
        if not np.any(mask):
            return 0
        
        # Histogram of directions
        directions = direction[mask]
        hist, _ = np.histogram(directions, bins=16, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-6)
        
        # Entropy as measure of directionality (lower = more directional)
        entropy = -np.sum(hist * np.log2(hist + 1e-6))
        
        # Normalize (max entropy for 16 bins is 4)
        directionality = 1 - (entropy / 4)
        
        return directionality
    
    def extract_gabor_features(self, image, scales=4, orientations=6):
        """
        Extract Gabor filter responses.
        
        Args:
            image: Input image
            scales: Number of scales
            orientations: Number of orientations
            
        Returns:
            Array of mean and std for each filter response
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float64)
        
        features = []
        
        for scale in range(scales):
            frequency = 0.1 + 0.1 * scale
            
            for orientation in range(orientations):
                theta = orientation * np.pi / orientations
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21),
                    sigma=4.0,
                    theta=theta,
                    lambd=1.0 / frequency,
                    gamma=0.5,
                    psi=0
                )
                
                # Apply filter
                response = cv2.filter2D(gray, cv2.CV_64F, kernel)
                
                # Extract statistics
                features.append(np.mean(np.abs(response)))
                features.append(np.std(response))
        
        features = np.array(features)
        
        # Normalize
        if features.max() > 0:
            features = features / features.max()
        
        return features
    
    def extract_lbp_features(self, image, radius=3, n_points=24):
        """
        Extract Local Binary Pattern histogram.
        
        Args:
            image: Input image
            radius: Radius of LBP
            n_points: Number of points for LBP
            
        Returns:
            Normalized LBP histogram
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate LBP
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate histogram
        n_bins = n_points + 2  # uniform LBP has P+2 patterns
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize
        hist = hist.astype(np.float64)
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist
    
    def extract_glcm_features(self, image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Extract Gray-Level Co-occurrence Matrix features.
        
        Returns:
            Dictionary with contrast, dissimilarity, homogeneity, energy, correlation
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Reduce levels for faster computation
        gray = (gray / 16).astype(np.uint8)
        
        # Compute GLCM
        glcm = graycomatrix(
            gray, 
            distances=distances, 
            angles=angles,
            levels=16,
            symmetric=True,
            normed=True
        )
        
        # Extract properties
        features = {
            'contrast': float(np.mean(graycoprops(glcm, 'contrast'))),
            'dissimilarity': float(np.mean(graycoprops(glcm, 'dissimilarity'))),
            'homogeneity': float(np.mean(graycoprops(glcm, 'homogeneity'))),
            'energy': float(np.mean(graycoprops(glcm, 'energy'))),
            'correlation': float(np.mean(graycoprops(glcm, 'correlation')))
        }
        
        return features
    
    # =====================
    # SHAPE DESCRIPTORS
    # =====================
    
    def extract_hu_moments(self, image):
        """
        Extract Hu moments (7 invariant moments).
        
        Returns:
            Array of 7 Hu moments (log transformed for scale)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform for scale invariance
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments
    
    def extract_contour_features(self, image):
        """
        Extract contour-based shape features.
        
        Returns:
            Dictionary with area, perimeter, circularity, solidity
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'area': 0,
                'perimeter': 0,
                'circularity': 0,
                'solidity': 0,
                'aspect_ratio': 1
            }
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 1
        
        # Normalize area and perimeter
        img_area = gray.shape[0] * gray.shape[1]
        
        return {
            'area': float(area / img_area),
            'perimeter': float(perimeter / np.sqrt(img_area)),
            'circularity': float(circularity),
            'solidity': float(solidity),
            'aspect_ratio': float(aspect_ratio)
        }
    
    def extract_hog_features(self, image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Extract Histogram of Oriented Gradients features.
        
        Returns:
            HOG feature vector
        """
        from skimage.feature import hog
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to fixed size for consistent feature length
        gray = cv2.resize(gray, (128, 128))
        
        # Extract HOG features
        features = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )
        
        return features
    
    # =====================
    # COMBINED EXTRACTION
    # =====================
    
    def extract_all_features(self, image):
        """
        Extract all visual features from an image.
        
        Returns:
            Dictionary with all features
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image")
        
        # Ensure minimum size
        if image.shape[0] < 10 or image.shape[1] < 10:
            image = cv2.resize(image, (64, 64))
        
        features = {}
        
        # Color features
        features['color_histogram_hsv'] = self.extract_color_histogram_hsv(image).tolist()
        features['dominant_colors'] = self.extract_dominant_colors(image, k=self.n_dominant_colors)
        features['color_moments'] = self.extract_color_moments(image).tolist()
        
        # Texture features
        features['tamura'] = self.extract_tamura_features(image)
        features['gabor'] = self.extract_gabor_features(
            image, 
            scales=self.gabor_scales, 
            orientations=self.gabor_orientations
        ).tolist()
        features['lbp'] = self.extract_lbp_features(
            image, 
            radius=self.lbp_radius, 
            n_points=self.lbp_points
        ).tolist()
        features['glcm'] = self.extract_glcm_features(image)
        
        # Shape features
        features['hu_moments'] = self.extract_hu_moments(image).tolist()
        features['contour'] = self.extract_contour_features(image)
        features['hog'] = self.extract_hog_features(image).tolist()
        
        return features
    
    def extract_feature_vector(self, image):
        """
        Extract a single concatenated feature vector for similarity search.
        
        Returns:
            numpy array of all features concatenated
        """
        features = self.extract_all_features(image)
        
        # Concatenate all numerical features
        vector = []
        
        # Color
        vector.extend(features['color_histogram_hsv'])
        vector.extend([c / 255.0 for color in features['dominant_colors']['colors'] for c in color])
        vector.extend(features['dominant_colors']['percentages'])
        vector.extend(features['color_moments'])
        
        # Texture
        vector.extend([
            features['tamura']['coarseness'],
            features['tamura']['contrast'],
            features['tamura']['directionality']
        ])
        vector.extend(features['gabor'])
        vector.extend(features['lbp'])
        vector.extend([
            features['glcm']['contrast'],
            features['glcm']['dissimilarity'],
            features['glcm']['homogeneity'],
            features['glcm']['energy'],
            features['glcm']['correlation']
        ])
        
        # Shape
        vector.extend(features['hu_moments'])
        vector.extend([
            features['contour']['area'],
            features['contour']['perimeter'],
            features['contour']['circularity'],
            features['contour']['solidity'],
            features['contour']['aspect_ratio']
        ])
        vector.extend(features['hog'])
        
        return np.array(vector)


# Singleton instance
feature_extractor = FeatureExtractor()
