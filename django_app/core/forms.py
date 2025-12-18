"""
Django Forms for CBIR Application
"""

from django import forms
from django.conf import settings


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


class ImageUploadForm(forms.Form):
    """Form for uploading images"""
    
    images = MultipleFileField(
        label='Select Images',
        help_text='You can select multiple images (JPG, PNG, GIF)',
        required=True
    )
    
    category = forms.ChoiceField(
        choices=[('', 'Auto-detect')] + [(c, c.title()) for c in settings.CATEGORY_LIST],
        required=False,
        label='Category'
    )
    
    auto_detect = forms.BooleanField(
        initial=True,
        required=False,
        label='Auto-detect objects after upload'
    )


class SearchForm(forms.Form):
    """Form for similarity search"""
    
    query_image = forms.ImageField(
        label='Query Image',
        required=True
    )
    
    top_k = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=10,
        label='Number of results'
    )
    
    metric = forms.ChoiceField(
        choices=[
            ('cosine', 'Cosine Distance'),
            ('euclidean', 'Euclidean Distance'),
            ('manhattan', 'Manhattan Distance'),
            ('chi_square', 'Chi-Square Distance'),
            ('intersection', 'Histogram Intersection'),
        ],
        initial='cosine',
        label='Distance Metric'
    )
    
    filter_class = forms.ChoiceField(
        choices=[('', 'All Classes')] + [(c, c.title()) for c in settings.CATEGORY_LIST],
        required=False,
        label='Filter by Class'
    )


class TransformForm(forms.Form):
    """Form for image transformations"""
    
    TRANSFORM_CHOICES = [
        ('crop', 'Crop'),
        ('resize', 'Resize'),
        ('rotate', 'Rotate'),
        ('flip', 'Flip'),
    ]
    
    transform_type = forms.ChoiceField(
        choices=TRANSFORM_CHOICES,
        label='Transform Type'
    )
    
    # Crop parameters
    x_min = forms.IntegerField(min_value=0, required=False, label='X Min')
    y_min = forms.IntegerField(min_value=0, required=False, label='Y Min')
    x_max = forms.IntegerField(min_value=0, required=False, label='X Max')
    y_max = forms.IntegerField(min_value=0, required=False, label='Y Max')
    
    # Resize parameters
    width = forms.IntegerField(min_value=1, required=False, label='Width')
    height = forms.IntegerField(min_value=1, required=False, label='Height')
    scale = forms.FloatField(min_value=0.01, max_value=10, required=False, label='Scale')
    
    # Rotate parameters
    angle = forms.FloatField(min_value=-360, max_value=360, required=False, label='Angle')
    
    # Flip parameters
    flip_direction = forms.ChoiceField(
        choices=[
            ('horizontal', 'Horizontal'),
            ('vertical', 'Vertical'),
            ('both', 'Both'),
        ],
        required=False,
        label='Flip Direction'
    )
    
    save_result = forms.BooleanField(
        initial=False,
        required=False,
        label='Save as new image'
    )


class DetectionSettingsForm(forms.Form):
    """Form for detection settings"""
    
    confidence = forms.FloatField(
        min_value=0.01,
        max_value=1.0,
        initial=0.25,
        label='Confidence Threshold'
    )
    
    iou = forms.FloatField(
        min_value=0.01,
        max_value=1.0,
        initial=0.45,
        label='IoU Threshold'
    )
