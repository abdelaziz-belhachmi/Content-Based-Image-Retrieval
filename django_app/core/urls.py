"""
URL patterns for core app
"""

from django.urls import path
from . import views
from . import views_3d

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
    path('images/bulk-delete/', views.BulkDeleteView.as_view(), name='bulk_delete'),
    path('images/delete-all/', views.DeleteAllView.as_view(), name='delete_all'),
    path('images/index-all/', views.IndexAllImagesView.as_view(), name='index_all'),
    
    # Detection
    path('image/<int:pk>/detect/', views.DetectObjectsView.as_view(), name='detect_objects'),
    
    # Descriptors
    path('object/<int:pk>/descriptors/', views.DescriptorDetailView.as_view(), name='descriptor_detail'),
    path('object/<int:pk>/extract/', views.ExtractDescriptorsView.as_view(), name='extract_descriptors'),
    
    # Search
    path('search/', views.SearchView.as_view(), name='search'),
    path('search/by-object/', views.ObjectSearchView.as_view(), name='search_by_object'),
    path('search/by-image/<int:pk>/', views.SearchByImageView.as_view(), name='search_by_image'),
    path('search/by-selected-objects/<int:pk>/', views.SearchBySelectedObjectsView.as_view(), name='search_by_selected_objects'),
    
    # Transform
    path('image/<int:pk>/transform/', views.TransformView.as_view(), name='transform'),
    
    # API Status
    path('api/status/', views.APIStatusView.as_view(), name='api_status'),
    
    # ==================== 3D Model Retrieval ====================
    # Gallery and browsing
    path('3d/', views_3d.Model3DGalleryView.as_view(), name='model3d_gallery'),
    path('3d/upload/', views_3d.Model3DUploadView.as_view(), name='model3d_upload'),
    path('3d/model/<str:model_id>/', views_3d.Model3DDetailView.as_view(), name='model3d_detail'),
    path('3d/model/<str:model_id>/descriptors/', views_3d.Model3DDescriptorsView.as_view(), name='model3d_descriptors'),
    path('3d/model/<str:model_id>/download/', views_3d.Model3DDownloadView.as_view(), name='model3d_download'),
    path('3d/model/<str:model_id>/delete/', views_3d.Model3DDeleteView.as_view(), name='model3d_delete'),
    
    # Search
    path('3d/search/', views_3d.Model3DSearchView.as_view(), name='model3d_search'),
    path('3d/search/by-model/<str:model_id>/', views_3d.Model3DSearchByIdView.as_view(), name='model3d_search_by_id'),
    
    # Index management
    path('3d/index/build/', views_3d.Model3DIndexBuildView.as_view(), name='model3d_index_build'),
    path('3d/index/clear/', views_3d.Model3DIndexClearView.as_view(), name='model3d_index_clear'),
    
    # Evaluation
    path('3d/evaluate/', views_3d.Model3DEvaluateView.as_view(), name='model3d_evaluate'),
]
