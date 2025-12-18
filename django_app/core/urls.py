"""
URL patterns for core app
"""

from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    # Home
    path('', views.HomeView.as_view(), name='home'),
    
    # Gallery
    path('gallery/', views.GalleryView.as_view(), name='gallery'),
    path('image/<int:pk>/', views.ImageDetailView.as_view(), name='image_detail'),
    
    # Upload & Delete
    path('upload/', views.ImageUploadView.as_view(), name='upload'),
    path('image/<int:pk>/delete/', views.ImageDeleteView.as_view(), name='image_delete'),
    
    # Detection
    path('image/<int:pk>/detect/', views.DetectObjectsView.as_view(), name='detect_objects'),
    
    # Descriptors
    path('object/<int:pk>/descriptors/', views.DescriptorDetailView.as_view(), name='descriptor_detail'),
    path('object/<int:pk>/extract/', views.ExtractDescriptorsView.as_view(), name='extract_descriptors'),
    
    # Search
    path('search/', views.SearchView.as_view(), name='search'),
    path('search/by-object/', views.ObjectSearchView.as_view(), name='search_by_object'),
    path('search/by-image/<int:pk>/', views.SearchByImageView.as_view(), name='search_by_image'),
    
    # Transform
    path('image/<int:pk>/transform/', views.TransformView.as_view(), name='transform'),
    
    # API Status
    path('api/status/', views.APIStatusView.as_view(), name='api_status'),
]
