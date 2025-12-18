from django.contrib import admin
from .models import Image, DetectedObject, Descriptor, SearchHistory


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['filename', 'category', 'width', 'height', 'uploaded_at', 'features_extracted']
    list_filter = ['category', 'features_extracted', 'uploaded_at']
    search_fields = ['filename', 'category']
    readonly_fields = ['uploaded_at', 'width', 'height', 'file_size']


@admin.register(DetectedObject)
class DetectedObjectAdmin(admin.ModelAdmin):
    list_display = ['image', 'class_name', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max']
    list_filter = ['class_name']
    search_fields = ['class_name', 'image__filename']


@admin.register(Descriptor)
class DescriptorAdmin(admin.ModelAdmin):
    list_display = ['detected_object', 'created_at', 'updated_at']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ['created_at', 'metric_used', 'num_results', 'search_time_ms']
    list_filter = ['metric_used', 'created_at']
    readonly_fields = ['created_at']
