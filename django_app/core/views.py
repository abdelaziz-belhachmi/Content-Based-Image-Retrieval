"""
Django Views for CBIR Application
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.views.generic import ListView, DetailView, TemplateView
from django.contrib import messages
from django.conf import settings
from django.core.paginator import Paginator
from django.db.models import Count
import json
import os

from .models import Image, DetectedObject, Descriptor, SearchHistory
from .forms import ImageUploadForm, SearchForm, TransformForm
from .api_client import api_client


class HomeView(TemplateView):
    """Home page with dashboard"""
    template_name = 'core/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Statistics
        context['total_images'] = Image.objects.count()
        context['total_objects'] = DetectedObject.objects.count()
        context['total_searches'] = SearchHistory.objects.count()
        
        # Recent images
        context['recent_images'] = Image.objects.all()[:6]
        
        # Category distribution
        context['categories'] = (
            DetectedObject.objects
            .values('class_name')
            .annotate(count=Count('id'))
            .order_by('-count')[:10]
        )
        
        # API status
        context['api_status'] = api_client.health_check()
        
        return context


class GalleryView(ListView):
    """Image gallery with pagination and filtering"""
    model = Image
    template_name = 'core/gallery.html'
    context_object_name = 'images'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Image.objects.all()
        
        # Filter by category
        category = self.request.GET.get('category')
        if category:
            queryset = queryset.filter(category=category)
        
        # Search by filename
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(filename__icontains=search)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Pass categories as list of tuples (value, display_name)
        context['categories'] = [(cat, cat.replace('_', ' ').title()) for cat in settings.CATEGORY_LIST]
        context['current_category'] = self.request.GET.get('category', '')
        context['search_query'] = self.request.GET.get('search', '')
        return context


class ImageDetailView(DetailView):
    """Detailed view of a single image with detected objects"""
    model = Image
    template_name = 'core/image_detail.html'
    context_object_name = 'image'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['detected_objects'] = self.object.detected_objects.all()
        # Get descriptors for all detected objects
        context['descriptors'] = Descriptor.objects.filter(
            detected_object__image=self.object
        )
        return context


class ImageUploadView(View):
    """Handle image uploads"""
    template_name = 'core/upload.html'
    
    def get(self, request):
        form = ImageUploadForm()
        return render(request, self.template_name, {'form': form})
    
    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            files = request.FILES.getlist('images')
            category = form.cleaned_data.get('category')
            auto_detect = form.cleaned_data.get('auto_detect', True)
            
            uploaded_count = 0
            errors = []
            
            for f in files:
                try:
                    # Create image record
                    image = Image.objects.create(
                        file=f,
                        filename=f.name,
                        category=category,
                        file_size=f.size
                    )
                    
                    # Get image dimensions
                    from PIL import Image as PILImage
                    pil_image = PILImage.open(image.file.path)
                    image.width, image.height = pil_image.size
                    image.save()
                    
                    # Auto-detect objects if requested
                    if auto_detect:
                        self._detect_objects(image)
                    
                    uploaded_count += 1
                    
                except Exception as e:
                    errors.append(f"{f.name}: {str(e)}")
            
            if uploaded_count > 0:
                messages.success(request, f'Successfully uploaded {uploaded_count} image(s).')
            if errors:
                messages.error(request, f'Errors: {"; ".join(errors)}')
            
            return redirect('core:gallery')
        else:
            # Show form errors to user
            for field, field_errors in form.errors.items():
                for error in field_errors:
                    messages.error(request, f'{field}: {error}')
        
        return render(request, self.template_name, {'form': form})
    
    def _detect_objects(self, image):
        """Run object detection on an image and index it for search"""
        try:
            result = api_client.detect_objects(image.file.path)
            
            if result.get('success') and 'data' in result:
                detections = result['data'].get('detections', [])
                best_detection = None
                
                for det in detections:
                    DetectedObject.objects.create(
                        image=image,
                        class_id=det['class_id'],
                        class_name=det['class_name'],
                        confidence=det['confidence'],
                        x_min=det['bbox']['x_min'],
                        y_min=det['bbox']['y_min'],
                        x_max=det['bbox']['x_max'],
                        y_max=det['bbox']['y_max']
                    )
                    # Track highest confidence detection for auto-category
                    if best_detection is None or det['confidence'] > best_detection['confidence']:
                        best_detection = det
                
                # Auto-assign category if not set
                if not image.category and best_detection:
                    image.category = best_detection['class_name']
                    image.save()
                
                # Add image to object-based similarity search index
                # This properly indexes detected objects with their features
                try:
                    index_result = api_client.add_to_object_index(
                        image_path=image.file.path,
                        image_id=image.id,
                        confidence=0.25
                    )
                    
                    # If indexing was successful and returned descriptors, save them
                    if index_result.get('success') and 'data' in index_result:
                        descriptors_data = index_result['data'].get('descriptors', [])
                        detected_objects = list(image.detected_objects.all())
                        
                        # Match descriptors to detected objects by class and bbox
                        for desc_info in descriptors_data:
                            for det_obj in detected_objects:
                                # Match by class_id and approximate bbox
                                if det_obj.class_id == desc_info.get('class_id'):
                                    desc_dict = desc_info.get('descriptors', {})
                                    Descriptor.objects.update_or_create(
                                        detected_object=det_obj,
                                        defaults={
                                            'dominant_colors': desc_dict.get('dominant_colors'),
                                            'color_moments': desc_dict.get('color_moments'),
                                            'tamura_features': desc_dict.get('tamura'),
                                            'glcm_features': desc_dict.get('glcm'),
                                            'hu_moments': desc_dict.get('hu_moments'),
                                            'contour_features': desc_dict.get('contour'),
                                        }
                                    )
                                    break
                        
                        print(f"[INDEX] Indexed image {image.id} with {len(descriptors_data)} objects")
                except Exception as e:
                    print(f"Indexing error: {e}")
                    
        except Exception as e:
            print(f"Detection error: {e}")


class ImageDeleteView(View):
    """Delete an image"""
    
    def post(self, request, pk):
        image = get_object_or_404(Image, pk=pk)
        
        # Delete file
        if image.file and os.path.exists(image.file.path):
            os.remove(image.file.path)
        
        # Delete database record
        image.delete()
        
        messages.success(request, 'Image deleted successfully.')
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True})
        
        return redirect('core:gallery')


class BulkDeleteView(View):
    """Bulk delete multiple images"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            image_ids = data.get('image_ids', [])
            
            if not image_ids:
                return JsonResponse({'success': False, 'error': 'No images selected'})
            
            deleted_count = 0
            for image_id in image_ids:
                try:
                    image = Image.objects.get(pk=image_id)
                    # Delete file
                    if image.file and os.path.exists(image.file.path):
                        os.remove(image.file.path)
                    # Delete database record
                    image.delete()
                    deleted_count += 1
                except Image.DoesNotExist:
                    continue
                except Exception as e:
                    print(f"Error deleting image {image_id}: {e}")
            
            return JsonResponse({
                'success': True,
                'deleted_count': deleted_count,
                'message': f'Successfully deleted {deleted_count} image(s)'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})


class DeleteAllView(View):
    """Delete all images"""
    
    def post(self, request):
        try:
            images = Image.objects.all()
            deleted_count = 0
            
            for image in images:
                try:
                    if image.file and os.path.exists(image.file.path):
                        os.remove(image.file.path)
                    image.delete()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting image {image.id}: {e}")
            
            messages.success(request, f'Successfully deleted {deleted_count} image(s).')
            return redirect('core:gallery')
            
        except Exception as e:
            messages.error(request, f'Error deleting images: {str(e)}')
            return redirect('core:gallery')


class IndexAllImagesView(View):
    """Index all images for similarity search"""
    
    def post(self, request):
        try:
            images = Image.objects.all()
            indexed_count = 0
            errors = []
            
            for image in images:
                try:
                    if image.file and os.path.exists(image.file.path):
                        # Use object-based indexing
                        result = api_client.add_to_object_index(
                            image_path=image.file.path,
                            image_id=image.id,
                            confidence=0.25
                        )
                        if result.get('success'):
                            indexed_count += 1
                            
                            # Save descriptors if returned
                            if 'data' in result:
                                descriptors_data = result['data'].get('descriptors', [])
                                detected_objects = list(image.detected_objects.all())
                                
                                for desc_info in descriptors_data:
                                    for det_obj in detected_objects:
                                        if det_obj.class_id == desc_info.get('class_id'):
                                            desc_dict = desc_info.get('descriptors', {})
                                            Descriptor.objects.update_or_create(
                                                detected_object=det_obj,
                                                defaults={
                                                    'dominant_colors': desc_dict.get('dominant_colors'),
                                                    'color_moments': desc_dict.get('color_moments'),
                                                    'tamura_features': desc_dict.get('tamura'),
                                                    'glcm_features': desc_dict.get('glcm'),
                                                    'hu_moments': desc_dict.get('hu_moments'),
                                                    'contour_features': desc_dict.get('contour'),
                                                }
                                            )
                                            break
                        else:
                            errors.append(f"Image {image.id}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    errors.append(f"Image {image.id}: {str(e)}")
            
            if indexed_count > 0:
                messages.success(request, f'Successfully indexed {indexed_count} image(s) for search.')
            if errors:
                messages.warning(request, f'Some images failed to index: {len(errors)} errors')
            
            return redirect('core:gallery')
            
        except Exception as e:
            messages.error(request, f'Error indexing images: {str(e)}')
            return redirect('core:gallery')


class DetectObjectsView(View):
    """Run object detection on an image"""
    
    def post(self, request, pk):
        image = get_object_or_404(Image, pk=pk)
        
        try:
            confidence = float(request.POST.get('confidence', 0.25))
            
            result = api_client.detect_objects(
                image.file.path,
                confidence=confidence,
                draw_boxes=True
            )
            
            if result.get('success'):
                # Clear existing detections
                image.detected_objects.all().delete()
                
                # Save new detections
                detections = result['data'].get('detections', [])
                best_detection = None
                
                for det in detections:
                    DetectedObject.objects.create(
                        image=image,
                        class_id=det['class_id'],
                        class_name=det['class_name'],
                        confidence=det['confidence'],
                        x_min=det['bbox']['x_min'],
                        y_min=det['bbox']['y_min'],
                        x_max=det['bbox']['x_max'],
                        y_max=det['bbox']['y_max']
                    )
                    # Track highest confidence detection
                    if best_detection is None or det['confidence'] > best_detection['confidence']:
                        best_detection = det
                
                # Auto-assign category if not set
                if not image.category and best_detection:
                    image.category = best_detection['class_name']
                    image.save()
                
                return JsonResponse({
                    'success': True,
                    'data': result['data']
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Detection failed')
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })


class ExtractDescriptorsView(View):
    """Extract feature descriptors for an object"""
    
    def post(self, request, pk):
        obj = get_object_or_404(DetectedObject, pk=pk)
        
        try:
            result = api_client.extract_object_descriptors(
                obj.image.file.path,
                obj.bbox
            )
            
            if result.get('success'):
                features = result['data']['features']
                
                # Save or update descriptor
                descriptor, created = Descriptor.objects.update_or_create(
                    detected_object=obj,
                    defaults={
                        'dominant_colors': features.get('dominant_colors'),
                        'color_moments': features.get('color_moments'),
                        'tamura_features': features.get('tamura'),
                        'glcm_features': features.get('glcm'),
                        'hu_moments': features.get('hu_moments'),
                        'contour_features': features.get('contour'),
                    }
                )
                
                return JsonResponse({
                    'success': True,
                    'data': {
                        'features': features,
                        'feature_vector_length': result['data'].get('feature_vector_length')
                    }
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Extraction failed')
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })


class SearchView(TemplateView):
    """Image similarity search"""
    template_name = 'core/search.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SearchForm()
        context['metrics'] = ['cosine', 'euclidean', 'manhattan', 'chi_square', 'intersection']
        context['categories'] = settings.CATEGORY_LIST
        return context
    
    def post(self, request):
        form = SearchForm(request.POST, request.FILES)

        if form.is_valid():
            query_image = request.FILES.get('query_image')
            top_k = form.cleaned_data.get('top_k', 10)
            metric = form.cleaned_data.get('metric', 'cosine')
            filter_class = form.cleaned_data.get('filter_class')
            confidence = float(request.POST.get('confidence', 0.25))

            try:
                # Save temporary query image
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    for chunk in query_image.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name

                # Use object-based search
                result = api_client.search_by_object(
                    tmp_path,
                    query_image_id=None,  # No image to exclude for uploaded query
                    top_k=top_k,
                    metric=metric,
                    confidence=confidence
                )

                # Clean up temp file
                os.unlink(tmp_path)

                # Check if search was successful
                if not result.get('success'):
                    messages.error(request, result.get('message', 'Search failed'))
                    context = self.get_context_data()
                    context['form'] = form
                    return render(request, self.template_name, context)

                # Enrich results with Django Image objects
                api_results = result.get('data', {}).get('results', [])
                query_classes = result.get('data', {}).get('query_classes', [])
                enriched_results = []
                
                for r in api_results:
                    try:
                        # image_id might be like "123" or "img_0_filename.jpg"
                        image_id_str = str(r.get('image_id', ''))
                        
                        # Try to extract numeric ID
                        if image_id_str.isdigit():
                            img = Image.objects.get(pk=int(image_id_str))
                        else:
                            # For indexed images like "img_0_filename.jpg", skip
                            continue
                        
                        enriched_results.append({
                            'image': img,
                            'similarity': r.get('similarity', 0) * 100,  # Convert to percentage
                            'distance': r.get('score', 0),
                            'matching_classes': r.get('matching_classes', []),
                            'num_matching_objects': r.get('num_matching_objects', 0)
                        })
                    except (Image.DoesNotExist, ValueError):
                        continue

                # Get evaluation metrics from API response
                evaluation_metrics = result.get('data', {}).get('evaluation_metrics', {})

                # Prepare context
                context = self.get_context_data()
                context['form'] = form
                context['search_performed'] = True
                context['results'] = enriched_results
                context['result_count'] = len(enriched_results)
                context['query_classes'] = query_classes
                context['metric'] = metric
                context['top_k'] = top_k
                context['evaluation_metrics'] = evaluation_metrics

                # Log search
                SearchHistory.objects.create(
                    metric_used=metric,
                    num_results=len(enriched_results)
                )

                # Render object-based results template
                return render(request, 'core/object_search_results.html', context)

            except Exception as e:
                messages.error(request, f'Search error: {str(e)}')
        else:
            # Show form validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')

        context = self.get_context_data()
        context['form'] = form
        return render(request, self.template_name, context)


class SearchByImageView(View):
    """Search for similar images using an existing image in the database"""
    
    def get(self, request, pk):
        image = get_object_or_404(Image, pk=pk)
        form = SearchForm()
        
        return render(request, 'core/search.html', {
            'form': form,
            'query_image': image,
            'metrics': ['cosine', 'euclidean', 'manhattan', 'chi_square', 'intersection'],
            'categories': settings.CATEGORY_LIST
        })
    
    def post(self, request, pk):
        image = get_object_or_404(Image, pk=pk)

        top_k = int(request.POST.get('top_k', 10))
        metric = request.POST.get('metric', 'cosine')
        filter_class = request.POST.get('filter_class')
        confidence = float(request.POST.get('confidence', 0.25))

        try:
            # Use object-based search, exclude query image from results
            result = api_client.search_by_object(
                image.file.path,
                query_image_id=image.id,  # Exclude this image from results
                top_k=top_k,
                metric=metric,
                confidence=confidence
            )

            # Log search
            SearchHistory.objects.create(
                query_image=image,
                metric_used=metric,
                num_results=len(result.get('data', {}).get('results', []))
            )

            # Check if search was successful
            if not result.get('success'):
                messages.error(request, result.get('message', 'Search failed'))
                return redirect('core:search_by_image', pk=pk)

            # Enrich results for template
            api_results = result.get('data', {}).get('results', [])
            query_classes = result.get('data', {}).get('query_classes', [])
            enriched_results = []
            
            for r in api_results:
                try:
                    # image_id might be like "123" or "img_0_filename.jpg"
                    image_id_str = str(r.get('image_id', ''))
                    
                    # Try to extract numeric ID
                    if image_id_str.isdigit():
                        img = Image.objects.get(pk=int(image_id_str))
                    else:
                        continue
                    
                    enriched_results.append({
                        'image': img,
                        'similarity': r.get('similarity', 0) * 100,
                        'distance': r.get('score', 0),
                        'matching_classes': r.get('matching_classes', []),
                        'num_matching_objects': r.get('num_matching_objects', 0)
                    })
                except (Image.DoesNotExist, ValueError):
                    continue

            # Get evaluation metrics from API response
            evaluation_metrics = result.get('data', {}).get('evaluation_metrics', {})

            return render(request, 'core/object_search_results.html', {
                'query_image': image,
                'results': enriched_results,
                'query_classes': query_classes,
                'metric': metric,
                'top_k': top_k,
                'evaluation_metrics': evaluation_metrics
            })

        except Exception as e:
            messages.error(request, f'Search error: {str(e)}')
            return redirect('core:search_by_image', pk=pk)


class SearchBySelectedObjectsView(View):
    """Search for similar images using specific selected objects from an image"""
    
    def post(self, request, pk):
        image = get_object_or_404(Image, pk=pk)
        
        # Get selected object IDs
        object_ids = request.POST.getlist('object_ids')
        top_k = int(request.POST.get('top_k', 10))
        metric = request.POST.get('metric', 'cosine')
        
        if not object_ids:
            messages.error(request, 'Please select at least one object to search.')
            return redirect('core:image_detail', pk=pk)
        
        try:
            # Get the selected detected objects from the database
            selected_objects = DetectedObject.objects.filter(
                id__in=object_ids,
                image=image
            )
            
            if not selected_objects.exists():
                messages.error(request, 'Selected objects not found.')
                return redirect('core:image_detail', pk=pk)
            
            # Build objects config for API call
            objects_config = []
            for obj in selected_objects:
                objects_config.append({
                    'class_name': obj.class_name,
                    'bbox': {
                        'x_min': obj.x_min,
                        'y_min': obj.y_min,
                        'x_max': obj.x_max,
                        'y_max': obj.y_max
                    }
                })
            
            # Call the API with selected objects
            result = api_client.search_by_selected_objects(
                image_path=image.file.path,
                objects=objects_config,
                query_image_id=image.id,
                top_k=top_k,
                metric=metric
            )
            
            # Log search
            SearchHistory.objects.create(
                query_image=image,
                metric_used=metric,
                num_results=len(result.get('data', {}).get('results', []))
            )
            
            # Check if search was successful
            if not result.get('success'):
                messages.error(request, result.get('message', 'Search failed'))
                return redirect('core:image_detail', pk=pk)
            
            # Enrich results for template
            api_results = result.get('data', {}).get('results', [])
            query_classes = result.get('data', {}).get('query_classes', [])
            enriched_results = []
            
            for r in api_results:
                try:
                    image_id_str = str(r.get('image_id', ''))
                    
                    if image_id_str.isdigit():
                        img = Image.objects.get(pk=int(image_id_str))
                    else:
                        continue
                    
                    enriched_results.append({
                        'image': img,
                        'similarity': r.get('similarity', 0) * 100,
                        'distance': r.get('score', 0),
                        'matching_classes': r.get('matching_classes', []),
                        'num_matching_objects': r.get('num_matching_objects', 0)
                    })
                except (Image.DoesNotExist, ValueError):
                    continue
            
            # Get evaluation metrics from API response
            evaluation_metrics = result.get('data', {}).get('evaluation_metrics', {})
            
            return render(request, 'core/object_search_results.html', {
                'query_image': image,
                'results': enriched_results,
                'query_classes': query_classes,
                'selected_objects': list(selected_objects),
                'metric': metric,
                'top_k': top_k,
                'evaluation_metrics': evaluation_metrics
            })
        
        except Exception as e:
            messages.error(request, f'Search error: {str(e)}')
            return redirect('core:image_detail', pk=pk)


class ObjectSearchView(View):
    """Search by specific object in an image"""
    
    def post(self, request):
        try:
            image_id = request.POST.get('image_id')
            top_k = int(request.POST.get('top_k', 10))
            metric = request.POST.get('metric', 'cosine')
            confidence = float(request.POST.get('confidence', 0.25))

            image = get_object_or_404(Image, pk=image_id)

            result = api_client.search_by_object(
                image.file.path,
                query_image_id=image.id,  # Exclude query image from results
                top_k=top_k,
                metric=metric,
                confidence=confidence
            )

            # Check if search was successful
            if not result.get('success'):
                messages.error(request, result.get('message', 'Search failed'))
                return redirect('core:search')

            # Prepare enriched results for template
            api_results = result.get('data', {}).get('results', [])
            query_classes = result.get('data', {}).get('query_classes', [])
            enriched_results = []
            
            for r in api_results:
                try:
                    image_id_str = str(r.get('image_id', ''))
                    
                    if image_id_str.isdigit():
                        img = Image.objects.get(pk=int(image_id_str))
                    else:
                        continue
                    
                    enriched_results.append({
                        'image': img,
                        'similarity': r.get('similarity', 0) * 100,
                        'distance': r.get('score', 0),
                        'matching_classes': r.get('matching_classes', []),
                        'num_matching_objects': r.get('num_matching_objects', 0)
                    })
                except (Image.DoesNotExist, ValueError):
                    continue

            # Query image URL
            query_image_url = image.file.url if hasattr(image.file, 'url') else ''

            # Get evaluation metrics from API response
            evaluation_metrics = result.get('data', {}).get('evaluation_metrics', {})

            context = {
                'query_image': image,
                'query_image_url': query_image_url,
                'results': enriched_results,
                'query_classes': query_classes,
                'descriptor_type': 'object',
                'distance_metric': metric,
                'evaluation_metrics': evaluation_metrics,
            }
            return render(request, 'core/object_search_results.html', context)

        except Exception as e:
            messages.error(request, f'Object search error: {str(e)}')
            return redirect('core:search')


class TransformView(View):
    """Image transformation view"""
    template_name = 'core/transform.html'
    
    def get(self, request, pk):
        image = get_object_or_404(Image, pk=pk)
        form = TransformForm()
        return render(request, self.template_name, {
            'image': image,
            'form': form
        })
    
    def post(self, request, pk):
        image = get_object_or_404(Image, pk=pk)
        form = TransformForm(request.POST)
        
        if form.is_valid():
            transform_type = form.cleaned_data['transform_type']
            save_result = form.cleaned_data.get('save_result', False)
            
            params = {}
            
            if transform_type == 'crop':
                params = {
                    'x_min': form.cleaned_data.get('x_min', 0),
                    'y_min': form.cleaned_data.get('y_min', 0),
                    'x_max': form.cleaned_data.get('x_max', image.width),
                    'y_max': form.cleaned_data.get('y_max', image.height),
                }
            elif transform_type == 'resize':
                if form.cleaned_data.get('scale'):
                    params['scale'] = form.cleaned_data['scale']
                else:
                    params['width'] = form.cleaned_data.get('width')
                    params['height'] = form.cleaned_data.get('height')
            elif transform_type == 'rotate':
                params['angle'] = form.cleaned_data.get('angle', 0)
            elif transform_type == 'flip':
                params['direction'] = form.cleaned_data.get('flip_direction', 'horizontal')
            
            result = api_client.transform_image(
                image.file.path,
                transform_type,
                params,
                save=save_result
            )
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse(result)
            
            if result.get('success'):
                messages.success(request, 'Image transformed successfully.')
                if save_result and result.get('data', {}).get('saved_path'):
                    # Create new image record
                    saved_path = result['data']['saved_path']
                    # Handle saved image...
            else:
                messages.error(request, result.get('error', 'Transform failed'))
        
        return render(request, self.template_name, {
            'image': image,
            'form': form
        })


class DescriptorDetailView(View):
    """View descriptor details for an object"""
    
    def get(self, request, pk):
        obj = get_object_or_404(DetectedObject, pk=pk)
        
        try:
            descriptor = obj.descriptor
            data = {
                'dominant_colors': descriptor.dominant_colors,
                'color_moments': descriptor.color_moments,
                'tamura_features': descriptor.tamura_features,
                'glcm_features': descriptor.glcm_features,
                'hu_moments': descriptor.hu_moments,
                'contour_features': descriptor.contour_features,
            }
            return JsonResponse({'success': True, 'data': data})
        except Descriptor.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Descriptors not yet extracted'
            })


class APIStatusView(View):
    """Check Flask API status"""
    
    def get(self, request):
        status = api_client.health_check()
        return JsonResponse(status)
