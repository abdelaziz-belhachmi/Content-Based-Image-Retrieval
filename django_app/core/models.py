"""
Django Models for CBIR Application
"""

from django.db import models
from django.utils import timezone
import os


def image_upload_path(instance, filename):
    """Generate upload path for images"""
    ext = filename.split('.')[-1]
    new_filename = f"{instance.category or 'uncategorized'}_{timezone.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    return os.path.join('uploads', new_filename)


class Image(models.Model):
    """Model for uploaded images"""
    
    file = models.ImageField(upload_to=image_upload_path)
    filename = models.CharField(max_length=255)
    category = models.CharField(max_length=100, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    file_size = models.IntegerField(default=0)  # in bytes
    
    # Feature extraction status
    features_extracted = models.BooleanField(default=False)
    feature_vector = models.BinaryField(null=True, blank=True)  # Serialized numpy array
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.filename} ({self.category or 'uncategorized'})"
    
    def save(self, *args, **kwargs):
        if self.file:
            self.filename = os.path.basename(self.file.name)
            if hasattr(self.file, 'size'):
                self.file_size = self.file.size
        super().save(*args, **kwargs)
    
    @property
    def file_url(self):
        if self.file:
            return self.file.url
        return None
    
    @property
    def image(self):
        """Alias for file field for template compatibility"""
        return self.file
    
    @property
    def original_filename(self):
        """Alias for filename for template compatibility"""
        return self.filename
    
    @property
    def is_processed(self):
        """Check if image has been processed (has detected objects)"""
        return self.detected_objects.exists()


class DetectedObject(models.Model):
    """Model for detected objects in images"""
    
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name='detected_objects')
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    
    # Bounding box coordinates
    x_min = models.IntegerField()
    y_min = models.IntegerField()
    x_max = models.IntegerField()
    y_max = models.IntegerField()
    
    # Feature extraction
    feature_vector = models.BinaryField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-confidence']
    
    def __str__(self):
        return f"{self.class_name} ({self.confidence:.2%}) in {self.image.filename}"
    
    @property
    def bbox(self):
        return {
            'x_min': self.x_min,
            'y_min': self.y_min,
            'x_max': self.x_max,
            'y_max': self.y_max
        }
    
    @property
    def bbox_x(self):
        return self.x_min
    
    @property
    def bbox_y(self):
        return self.y_min
    
    @property
    def bbox_width(self):
        return self.x_max - self.x_min
    
    @property
    def bbox_height(self):
        return self.y_max - self.y_min
    
    @property
    def color(self):
        """Return a color based on class_id for bounding box visualization"""
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8B500', '#00CED1', '#FF69B4', '#32CD32', '#FF4500'
        ]
        return colors[self.class_id % len(colors)]
    
    @property
    def width(self):
        return self.x_max - self.x_min
    
    @property
    def height(self):
        return self.y_max - self.y_min


class Descriptor(models.Model):
    """Model for storing computed descriptors"""
    
    detected_object = models.OneToOneField(
        DetectedObject, 
        on_delete=models.CASCADE, 
        related_name='descriptor'
    )
    
    # Color descriptors
    color_histogram = models.BinaryField(null=True, blank=True)
    dominant_colors = models.JSONField(null=True, blank=True)
    color_moments = models.JSONField(null=True, blank=True)
    
    # Texture descriptors
    tamura_features = models.JSONField(null=True, blank=True)
    gabor_features = models.BinaryField(null=True, blank=True)
    lbp_histogram = models.BinaryField(null=True, blank=True)
    glcm_features = models.JSONField(null=True, blank=True)
    
    # Shape descriptors
    hu_moments = models.JSONField(null=True, blank=True)
    contour_features = models.JSONField(null=True, blank=True)
    hog_features = models.BinaryField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Descriptors for {self.detected_object}"


class SearchHistory(models.Model):
    """Model for tracking search history"""
    
    query_image = models.ForeignKey(
        Image, 
        on_delete=models.SET_NULL, 
        null=True, 
        related_name='searches'
    )
    query_object = models.ForeignKey(
        DetectedObject,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='searches'
    )
    
    metric_used = models.CharField(max_length=50, default='cosine')
    num_results = models.IntegerField(default=0)
    search_time_ms = models.FloatField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Search histories'
    
    def __str__(self):
        return f"Search on {self.created_at.strftime('%Y-%m-%d %H:%M')}"
