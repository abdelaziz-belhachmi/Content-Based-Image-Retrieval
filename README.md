# CBIR System - Content-Based Image Retrieval with Object Detection

A Content-Based Image Retrieval (CBIR) system that combines YOLOv8 object detection with advanced visual feature descriptors for intelligent image search.

## Overview

This system allows users to:
- Upload and manage images in a gallery
- Automatically detect objects using a fine-tuned YOLOv8 model
- Extract visual descriptors (color, texture, shape) from detected objects
- Search for similar images based on visual content
- Apply image transformations

## Project Structure

```
Content-Based-Image-Retrieval/
|-- Data/                      # ImageNet dataset (15 categories)
|   |-- Ananas/
|   |-- Apple/
|   |-- BellPepper/
|   +-- ... (15 categories total)
|-- yolo_training/             # YOLO model training
|   |-- dataset/               # YOLO format dataset
|   |-- models/                # Trained models
|   |-- data.yaml              # Dataset configuration
|   |-- convert_annotations.py # VOC to YOLO converter
|   |-- fix_corrupt_images.py  # Image validation
|   +-- train.py               # Training script
|-- flask_api/                 # Flask REST API
|   |-- app.py                 # Main application
|   |-- config.py              # Configuration
|   |-- services/              # Core services
|   |   |-- detection.py       # YOLOv8 detection
|   |   |-- descriptors.py     # Feature extraction
|   |   |-- similarity.py      # Legacy similarity search
|   |   |-- similarity_objects.py  # Object-based search
|   |   +-- object_index_persistence.py  # Index persistence
|   +-- resources/             # API endpoints
|       |-- detection.py
|       |-- descriptors.py
|       |-- images.py
|       |-- search.py
|       +-- search_objects.py
|-- django_app/                # Django web application
|   |-- config/                # Django settings
|   |-- core/                  # Main app
|   |   |-- models.py          # Database models
|   |   |-- views.py           # View handlers
|   |   |-- api_client.py      # Flask API client
|   |   +-- forms.py           # Form definitions
|   |-- templates/             # HTML templates
|   |-- static/                # CSS, JS files
|   +-- media/                 # Uploaded images
|-- filter_data.py             # Dataset filtering script
|-- requirements.txt           # Python dependencies
|-- cahier_de_charge.md        # Project specifications (French)
+-- README.md                  # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset (Optional - for training)

Convert ImageNet annotations to YOLO format:

```bash
cd yolo_training
python convert_annotations.py
```

### 4. Train YOLO Model (Optional)

Skip this step if using the pre-trained model in yolo_training/models/.

```bash
cd yolo_training
python train.py
```

### 5. Start Flask API

```bash
cd flask_api
python app.py
```

The API runs at: http://localhost:5000

### 6. Setup Django Application

```bash
cd django_app
python manage.py migrate
python manage.py createsuperuser  # Optional - for admin access
python manage.py runserver
```

The web application runs at: http://localhost:8000

## Features

### Object Detection

- Model: YOLOv8n (nano) fine-tuned on 15 classes
- Classes: pineapple, apple, bell_pepper, bicycle, broccoli, bus, car, cat, dog, elephant, horse, lemon, motorcycle, strawberry, tomato
- Real-time detection with bounding boxes and confidence scores

### Feature Descriptors

Color Descriptors:
- HSV Histograms - Color distribution in HSV space
- Dominant Colors - K-means clustering (k=5)
- Color Moments - Mean, standard deviation, skewness per channel

Texture Descriptors:
- Tamura Features - Coarseness, contrast, directionality
- Gabor Filters - 4 scales x 6 orientations
- LBP (Local Binary Pattern) - Local texture patterns
- GLCM - Gray-Level Co-occurrence Matrix features

Shape Descriptors:
- Hu Moments - 7 rotation/scale invariant moments
- HOG - Histogram of Oriented Gradients
- Contour Features - Perimeter, area, circularity, solidity

### Object-Based Similarity Search

The search system works at the object level:
1. Query image is analyzed to detect objects
2. Features are extracted for each detected object
3. Search finds images containing similar objects of the same class
4. Results are ranked by visual similarity

Key features:
- Persistent index (survives server restarts)
- Object-class filtering (horse query finds only horse images)
- Multiple distance metrics: Cosine, Euclidean, Manhattan, Chi-Square, Histogram Intersection
- Automatic indexing on image upload

### CBIR Evaluation Metrics

The system includes standard Information Retrieval metrics to evaluate search quality:

| Metric | Description | Usage |
|--------|-------------|-------|
| **P@K** (Precision@K) | Proportion of relevant items in top-K results | `precision_at_k(results, query_class, k=10)` |
| **mAP@K** (Mean Average Precision) | Average of AP@K across multiple queries | `mean_average_precision_at_k(query_results_list, k=50)` |
| **NDCG@K** (Normalized DCG) | Ranking quality with position-based discount | `ndcg_at_k(results, query_class, k=10)` |

**By Image** (`services/similarity.py`):
```python
from services.similarity import similarity_service

# Evaluate single search
metrics = similarity_service.evaluate_search(
    query_vector=features,
    query_class='Dog',
    metric='cosine'
)
# Returns: {'precision_at_10': 0.8, 'ap_at_50': 0.75, 'ndcg_at_10': 0.82, ...}

# Batch evaluation
batch_metrics = similarity_service.batch_evaluate(queries)
# Returns: {'mean_precision_at_10': 0.78, 'map_at_50': 0.72, 'mean_ndcg_at_10': 0.80}
```

**By Objects** (`services/similarity_objects.py`):
```python
from services.similarity_objects import object_similarity_service

# Evaluate object search
metrics = object_similarity_service.evaluate_object_search(
    query_vector=features,
    query_class='bus',
    metric='cosine'
)

# Evaluate multi-object image search
metrics = object_similarity_service.evaluate_image_objects_search(
    query_objects=[{'class_name': 'bus', 'feature_vector': [...]}],
    metric='cosine'
)
# Returns: {'precision_at_10': 0.7, 'ap_at_50': 0.68, 'ndcg_at_10': 0.75, ...}
```

### Image Transformations

- Crop - Extract regions of interest
- Resize - Scale images
- Rotate - Rotate by any angle
- Flip - Horizontal/vertical flip

## API Endpoints

### Detection

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/detect | Detect objects in an image |
| POST | /api/detect/batch | Batch detection for multiple images |

### Descriptors

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/descriptors/extract | Extract features from image |
| POST | /api/descriptors/extract/object | Extract features from cropped region |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/search/by-object | Search by detected objects (primary) |
| POST | /api/search/similar | Legacy similarity search |

### Index Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/index/add | Add single image to index |
| POST | /api/index/build | Build index from multiple images |
| GET | /api/index/stats | Get index statistics |
| DELETE | /api/index/clear | Clear the index |

### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/images/upload | Upload image |
| GET | /api/images/{id} | Get image details |
| DELETE | /api/images/{id} | Delete image |

### Transformations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/transform/crop | Crop image |
| POST | /api/transform/resize | Resize image |
| POST | /api/transform/rotate | Rotate image |
| POST | /api/transform/flip | Flip image |

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | API health status with index stats |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8n (Ultralytics) |
| Backend API | Flask + Flask-RESTful |
| Web Application | Django 5.0 |
| Frontend | Bootstrap 5 + JavaScript |
| Database | SQLite (development) |
| Image Processing | OpenCV, Pillow, scikit-image |
| Feature Extraction | NumPy, SciPy, scikit-learn |

## Dataset

Source: ImageNet (15 selected categories)

| Category | Images |
|----------|--------|
| Ananas (Pineapple) | 414 |
| Apple | 505 |
| Bell Pepper | 438 |
| Bicycle | 229 |
| Broccoli | 319 |
| Bus | 241 |
| Car | 710 |
| Cat | 381 |
| Dog | 609 |
| Elephant | 449 |
| Horse | 562 |
| Lemon | 429 |
| Motorcycle | 354 |
| Strawberry | 363 |
| Tomato | 89 |
| Total | 6,092 |

## Usage

1. Start both Flask API and Django server
2. Access the web interface at http://localhost:8000
3. Upload images through the Upload page
4. View images in the Gallery
5. Click on an image to see detected objects and descriptors
6. Use "Find Similar" to search for similar images
7. Adjust search parameters (metric, number of results)

