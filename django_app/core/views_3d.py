"""
Django Views for 3D Model Retrieval

Provides views for uploading, viewing, and searching 3D models.
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views import View
from django.views.generic import ListView, DetailView, TemplateView
from django.contrib import messages
from django.conf import settings
from django.core.paginator import Paginator
from django.db.models import Count
import json
import os
from pathlib import Path
import uuid

from .api_client import api_client


class Model3DGalleryView(TemplateView):
    """Gallery view for 3D models"""
    template_name = 'core/model3d_gallery.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get pagination parameters
        page = self.request.GET.get('page', 1)
        try:
            page = int(page)
        except ValueError:
            page = 1
        per_page = 12
        offset = (page - 1) * per_page
        
        # Get selected category filter
        category_filter = self.request.GET.get('category', '')
        
        # Get index stats from API
        try:
            response = api_client.get('/models3d/index')
            if response and response.get('success'):
                stats = response.get('stats', {})
                context['stats'] = stats
                context['categories'] = stats.get('categories', {})
                context['total_models'] = stats.get('total_models', 0)
                
                # Convert categories dict to list for template
                category_list = [
                    {'name': cat, 'count': count}
                    for cat, count in stats.get('categories', {}).items()
                ]
                context['category_list'] = sorted(category_list, key=lambda x: x['name'])
            else:
                context['stats'] = {}
                context['categories'] = {}
                context['total_models'] = 0
                context['category_list'] = []
                context['error'] = response.get('error', 'Failed to fetch index')
        except Exception as e:
            context['stats'] = {}
            context['error'] = str(e)
            context['categories'] = {}
            context['total_models'] = 0
            context['category_list'] = []
        
        # Get list of models with pagination
        try:
            params = {'limit': per_page, 'offset': offset}
            if category_filter:
                params['category'] = category_filter
            
            list_response = api_client.get('/models3d/list', params=params)
            if list_response and list_response.get('success'):
                context['models'] = list_response.get('models', [])
                context['models_total'] = list_response.get('total', 0)
                
                # Calculate pagination
                total_pages = (list_response.get('total', 0) + per_page - 1) // per_page
                context['page'] = page
                context['total_pages'] = total_pages
                context['has_prev'] = page > 1
                context['has_next'] = page < total_pages
                context['page_range'] = range(max(1, page - 2), min(total_pages + 1, page + 3))
            else:
                context['models'] = []
                context['models_total'] = 0
        except Exception as e:
            context['models'] = []
            context['models_total'] = 0
            
        context['title'] = 'Galerie de Modèles 3D'
        context['category_filter'] = category_filter
        return context


class Model3DUploadView(View):
    """Upload 3D models"""
    template_name = 'core/model3d_upload.html'
    
    def get(self, request):
        return render(request, self.template_name, {
            'title': 'Upload Modèle 3D'
        })
    
    def post(self, request):
        if 'file' not in request.FILES:
            messages.error(request, 'Aucun fichier sélectionné')
            return redirect('core:model3d_upload')
        
        uploaded_file = request.FILES['file']
        category = request.POST.get('category', 'unknown')
        
        # Check extension
        if not uploaded_file.name.lower().endswith('.obj'):
            messages.error(request, 'Seuls les fichiers OBJ sont acceptés')
            return redirect('core:model3d_upload')
        
        try:
            # Upload to API
            files = {'file': (uploaded_file.name, uploaded_file.read())}
            data = {'category': category}
            
            response = api_client.upload_file('/models3d/upload', files, data)
            
            if response and response.get('success'):
                model_id = response.get('model_id')
                filepath = response.get('filepath')
                
                # Index the model
                index_response = api_client.post('/models3d/index', {
                    'model_id': model_id,
                    'filepath': filepath,
                    'category': category
                })
                
                if index_response and index_response.get('success'):
                    messages.success(request, f'Modèle {uploaded_file.name} uploadé et indexé avec succès!')
                else:
                    messages.warning(request, f'Modèle uploadé mais indexation échouée')
                
                return redirect('core:model3d_detail', model_id=model_id)
            else:
                messages.error(request, response.get('error', 'Erreur lors de l\'upload'))
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
        
        return redirect('core:model3d_upload')


class Model3DDetailView(View):
    """Detail view for a 3D model"""
    template_name = 'core/model3d_detail.html'
    
    def get(self, request, model_id):
        try:
            response = api_client.get(f'/models3d/{model_id}')
            
            if response and response.get('success'):
                context = {
                    'model': response,
                    'model_id': model_id,
                    'title': f'Modèle 3D: {model_id}'
                }
                return render(request, self.template_name, context)
            else:
                messages.error(request, 'Modèle non trouvé')
                return redirect('core:model3d_gallery')
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
            return redirect('core:model3d_gallery')


class Model3DSearchView(View):
    """Search for similar 3D models"""
    template_name = 'core/model3d_search.html'
    
    def get(self, request):
        # Get available categories
        try:
            response = api_client.get('/models3d/index')
            categories = []
            if response and response.get('success'):
                categories = list(response.get('stats', {}).get('categories', {}).keys())
        except:
            categories = []
        
        context = {
            'title': 'Recherche de Modèles 3D Similaires',
            'categories': categories,
            'metrics': ['cosine', 'euclidean', 'manhattan', 'correlation']
        }
        return render(request, self.template_name, context)
    
    def post(self, request):
        # Handle file upload search
        if 'file' not in request.FILES:
            messages.error(request, 'Aucun fichier requête sélectionné')
            return redirect('core:model3d_search')
        
        uploaded_file = request.FILES['file']
        metric = request.POST.get('metric', 'cosine')
        k = int(request.POST.get('k', 10))
        category_filter = request.POST.get('category_filter', '') or None
        
        try:
            # Save file temporarily
            temp_dir = Path(settings.MEDIA_ROOT) / 'temp_3d'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"query_{uuid.uuid4()}.obj"
            
            with open(temp_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            # Search via API
            search_data = {
                'filepath': str(temp_path),
                'metric': metric,
                'k': k
            }
            if category_filter:
                search_data['category_filter'] = category_filter
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"3D Search request: {search_data}")
            
            response = api_client.post('/models3d/search', search_data)
            
            logger.info(f"3D Search response: {response}")
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if response and response.get('success'):
                context = {
                    'title': 'Résultats de Recherche 3D',
                    'query_filename': uploaded_file.name,
                    'results': response.get('results', []),
                    'count': response.get('count', 0),
                    'metric': metric,
                    'k': k,
                    'evaluation_metrics': response.get('evaluation_metrics'),
                    'query_category': response.get('query', {}).get('category')
                }
                return render(request, 'core/model3d_search_results.html', context)
            else:
                error_msg = response.get('error', 'Erreur de recherche inconnue') if response else 'Pas de réponse du serveur API'
                messages.error(request, f'Erreur: {error_msg}')
                
        except Exception as e:
            import traceback
            messages.error(request, f'Erreur: {str(e)}')
            logging.getLogger(__name__).error(f"3D Search error: {traceback.format_exc()}")
        
        return redirect('core:model3d_search')


class Model3DSearchByIdView(View):
    """Search using an indexed model as query"""
    
    def get(self, request, model_id):
        metric = request.GET.get('metric', 'cosine')
        k = int(request.GET.get('k', 10))
        
        try:
            response = api_client.post('/models3d/search', {
                'model_id': model_id,
                'metric': metric,
                'k': k
            })
            
            if response and response.get('success'):
                context = {
                    'title': 'Résultats de Recherche 3D',
                    'query_model_id': model_id,
                    'results': response.get('results', []),
                    'count': response.get('count', 0),
                    'metric': metric,
                    'k': k,
                    'evaluation_metrics': response.get('evaluation_metrics'),
                    'query_category': response.get('query', {}).get('category')
                }
                return render(request, 'core/model3d_search_results.html', context)
            else:
                messages.error(request, response.get('error', 'Erreur de recherche'))
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
        
        return redirect('core:model3d_detail', model_id=model_id)


class Model3DIndexBuildView(View):
    """Build index from directory"""
    template_name = 'core/model3d_index_build.html'
    
    # Default 3D data path
    DEFAULT_3D_PATH = r'E:\Content-Based-Image-Retrieval\Data_3D\models'
    
    def get(self, request):
        return render(request, self.template_name, {
            'title': 'Construire l\'Index 3D',
            'default_path': self.DEFAULT_3D_PATH
        })
    
    def post(self, request):
        directory = request.POST.get('directory', '') or self.DEFAULT_3D_PATH
        category = request.POST.get('category', 'unknown')
        
        if not directory:
            messages.error(request, 'Veuillez spécifier un répertoire')
            return redirect('core:model3d_index_build')
        
        try:
            response = api_client.post('/models3d/index/build', {
                'directory': directory,
                'category': category
            })
            
            if response and response.get('success'):
                indexed = response.get('indexed', 0)
                errors = response.get('errors', 0)
                messages.success(request, f'{indexed} modèles indexés, {errors} erreurs')
            else:
                messages.error(request, response.get('error', 'Erreur lors de l\'indexation'))
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
        
        return redirect('core:model3d_gallery')


class Model3DIndexClearView(View):
    """Clear the 3D model index"""
    
    def post(self, request):
        try:
            response = api_client.delete('/models3d/index')
            
            if response and response.get('success'):
                messages.success(request, 'Index 3D vidé avec succès')
            else:
                messages.error(request, 'Erreur lors du vidage de l\'index')
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
        
        return redirect('core:model3d_gallery')


class Model3DQuickIndexView(View):
    """Quick index using default Data_3D path from Flask config"""
    
    def post(self, request):
        try:
            # Use the Flask API's quick index endpoint which uses config paths
            response = api_client.post('/models3d/index/quick', {})
            
            if response and response.get('success'):
                indexed = response.get('indexed', 0)
                errors = response.get('errors', 0)
                default_path = response.get('default_path', '')
                messages.success(
                    request, 
                    f'Indexation terminée: {indexed} modèles indexés, {errors} erreurs (depuis {default_path})'
                )
            else:
                messages.error(request, response.get('error', 'Erreur lors de l\'indexation'))
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
        
        return redirect('core:model3d_gallery')


class Model3DDeleteView(View):
    """Delete a 3D model from index"""
    
    def post(self, request, model_id):
        try:
            response = api_client.delete(f'/models3d/{model_id}')
            
            if response and response.get('success'):
                messages.success(request, f'Modèle {model_id} supprimé')
            else:
                messages.error(request, 'Erreur lors de la suppression')
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
        
        return redirect('core:model3d_gallery')


class Model3DDescriptorsView(View):
    """View descriptors for a 3D model"""
    template_name = 'core/model3d_descriptors.html'
    
    def get(self, request, model_id):
        try:
            # Get model info
            model_response = api_client.get(f'/models3d/{model_id}')
            
            if not model_response or not model_response.get('success'):
                messages.error(request, 'Modèle non trouvé')
                return redirect('core:model3d_gallery')
            
            context = {
                'model': model_response,
                'model_id': model_id,
                'title': f'Descripteurs: {model_id}'
            }
            return render(request, self.template_name, context)
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
            return redirect('core:model3d_gallery')


class Model3DDownloadView(View):
    """Download a 3D model file"""
    
    def get(self, request, model_id):
        try:
            # Get model info to find filepath
            response = api_client.get(f'/models3d/{model_id}')
            
            if response and response.get('success'):
                filepath = response.get('filepath')
                if filepath and os.path.exists(filepath):
                    return FileResponse(
                        open(filepath, 'rb'),
                        as_attachment=True,
                        filename=os.path.basename(filepath)
                    )
            
            messages.error(request, 'Fichier non trouvé')
            
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
        
        return redirect('core:model3d_gallery')


class Model3DObjContentView(View):
    """Serve OBJ file content for Three.js viewer (inline, not as attachment)"""
    
    def get(self, request, model_id):
        try:
            # Get model info to find filepath
            api_response = api_client.get(f'/models3d/{model_id}')
            
            if not api_response:
                return HttpResponse(f'API returned no response for model {model_id}', status=500)
            
            if not api_response.get('success'):
                error_msg = api_response.get('error', 'Unknown error')
                return HttpResponse(f'API error: {error_msg}', status=404)
            
            filepath = api_response.get('filepath')
            if not filepath:
                return HttpResponse(f'No filepath in API response for model {model_id}', status=404)
            
            if not os.path.exists(filepath):
                return HttpResponse(f'File not found at path: {filepath}', status=404)
            
            # Serve file with proper content type for Three.js
            file_response = FileResponse(
                open(filepath, 'rb'),
                content_type='text/plain'
            )
            # Allow CORS for Three.js loader
            file_response['Access-Control-Allow-Origin'] = '*'
            file_response['Cache-Control'] = 'public, max-age=3600'
            return file_response
            
        except Exception as e:
            import traceback
            return HttpResponse(f'Error: {str(e)}\n{traceback.format_exc()}', status=500)


class Model3DEvaluateView(View):
    """Evaluate search performance"""
    template_name = 'core/model3d_evaluate.html'
    
    def get(self, request):
        return render(request, self.template_name, {
            'title': 'Évaluation des Performances'
        })
    
    def post(self, request):
        # Handle evaluation request
        if 'file' not in request.FILES:
            messages.error(request, 'Aucun fichier sélectionné')
            return redirect('core:model3d_evaluate')
        
        uploaded_file = request.FILES['file']
        category = request.POST.get('category', '')
        metric = request.POST.get('metric', 'cosine')
        
        if not category:
            messages.error(request, 'Veuillez spécifier la catégorie ground truth')
            return redirect('core:model3d_evaluate')
        
        try:
            # Save file temporarily
            temp_dir = Path(settings.MEDIA_ROOT) / 'temp_3d'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"eval_{uuid.uuid4()}.obj"
            
            with open(temp_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            # Evaluate via API
            response = api_client.post('/models3d/evaluate', {
                'filepath': str(temp_path),
                'category': category,
                'metric': metric
            })
            
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
            
            if response and response.get('success'):
                context = {
                    'title': 'Résultats d\'Évaluation',
                    'metrics': response.get('metrics', {}),
                    'query_filename': uploaded_file.name,
                    'category': category
                }
                return render(request, 'core/model3d_evaluate_results.html', context)
            else:
                messages.error(request, response.get('error', 'Erreur d\'évaluation'))
                
        except Exception as e:
            messages.error(request, f'Erreur: {str(e)}')
        
        return redirect('core:model3d_evaluate')
