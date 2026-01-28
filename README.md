# CBIR System - Content-Based Image & 3D Model Retrieval

A comprehensive Content-Based Information Retrieval (CBIR) system that combines:
- **2D Images**: YOLOv8 object detection with visual feature descriptors
- **3D Models**: Local feature-based retrieval using Spin Images, Shape Contexts, and PFH

## Overview

This system allows users to:
- Upload and manage images in a gallery
- Automatically detect objects using a fine-tuned YOLOv8 model
- Extract visual descriptors (color, texture, shape) from detected objects
- Search for similar images based on visual content
- Apply image transformations
- **Upload and index 3D models (OBJ format)**
- **Extract 3D local feature descriptors**
- **Search for similar 3D models by content**
- **Evaluate 3D retrieval performance**

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
|   |   |-- object_index_persistence.py  # Index persistence
|   |   |-- mesh_loader.py     # 3D OBJ file loader
|   |   |-- descriptors_3d.py  # 3D local feature extraction
|   |   +-- similarity_3d.py   # 3D model similarity search
|   +-- resources/             # API endpoints
|       |-- detection.py
|       |-- descriptors.py
|       |-- images.py
|       |-- search.py
|       |-- search_objects.py
|       +-- models_3d.py       # 3D model endpoints
|-- django_app/                # Django web application
|   |-- config/                # Django settings
|   |-- core/                  # Main app
|   |   |-- models.py          # Database models
|   |   |-- views.py           # View handlers (2D)
|   |   |-- views_3d.py        # View handlers (3D)
|   |   |-- api_client.py      # Flask API client
|   |   +-- forms.py           # Form definitions
|   |-- templates/             # HTML templates
|   |   |-- core/
|   |       |-- base.html, home.html, gallery.html, ...
|   |       |-- model3d_gallery.html    # 3D gallery
|   |       |-- model3d_upload.html     # 3D upload
|   |       |-- model3d_detail.html     # 3D model detail
|   |       |-- model3d_search.html     # 3D search form
|   |       +-- model3d_*.html          # Other 3D templates
|   |-- static/                # CSS, JS files
|   +-- media/                 # Uploaded images
|-- rapport_latex/             # LaTeX report
|   +-- rapport_3d_retrieval.tex
|-- Data_3D/                   # 3D benchmark data (after download)
|   |-- models/                # OBJ files
|   +-- thumbnails/            # Preview images
|-- filter_data.py             # Dataset filtering script
|-- download_3d_benchmark.py   # 3D dataset downloader
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

### 5. Download 3D Benchmark Dataset (For 3D Retrieval)

Download the 3D Pottery Benchmark Dataset:

```bash
python download_3d_benchmark.py
```

This script will:
- Download the 3D Pottery Dataset (1012 OBJ models)
- Extract and organize files into `Data_3D/models/`
- Prepare thumbnails in `Data_3D/thumbnails/`

### 5b. Index 3D Models (Optional - Command Line)

You can pre-index 3D models from the command line:

```bash
# Index all models
python index_3d_models.py

# Index with a limit (for testing)
python index_3d_models.py --limit 100

# Index and evaluate
python index_3d_models.py --evaluate --metric cosine

# Show index statistics
python index_3d_models.py --stats

# Clear the index
python index_3d_models.py --clear
```

Alternatively, use the web interface: **Recherche 3D > Construire Index**

### 6. Start Flask API

```bash
cd flask_api
python app.py
```

The API runs at: http://localhost:5000

### 7. Setup Django Application

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

## 3D Model Retrieval

### Overview

The 3D retrieval module implements content-based search for 3D models using **local feature descriptors**. Based on the research survey "A survey of content based 3D shape retrieval methods", this system focuses on local features for their robustness to partial matching and occlusion.

### Supported File Format

- **Wavefront OBJ (.obj)** - Standard 3D mesh format with vertices and faces

### 3D Feature Descriptors

| Descriptor | Type | Dimensions | Description |
|------------|------|------------|-------------|
| **Spin Images** | 2D Histogram | 256 (16×16) | Distribution of points in cylindrical coordinates around surface normals |
| **Shape Context 3D** | Spherical Histogram | 360 (5×6×12) | Spatial distribution in spherical bins (radial × azimuthal × elevation) |
| **Shape Index** | Curvature Histogram | 64 | Distribution of surface curvature types (convex, concave, saddle) |
| **PFH** | Angular Histogram | 33 (3×11) | Point Feature Histogram encoding angular relations between normals |

### Combined Descriptor

All local descriptors are concatenated and L2-normalized to create a unified feature vector of **713 dimensions**:
- Captures both geometric (PFH, Shape Index) and structural (Spin Image, Shape Context) information
- Enables single-vector similarity comparison

### Distance Metrics

- **Cosine** (recommended) - Angle between vectors, robust to magnitude
- **Euclidean** - L2 distance
- **Manhattan** - L1 distance
- **Correlation** - Pearson correlation coefficient

### 3D Benchmark Database

**3D Pottery Database** (http://www.ipet.gr/~akoutsou/benchmark/)
- 1012 pottery models in OBJ format
- Categories: Amphora, Alabastron, Hydria, Krater, Kylix, Lekythos, Oinochoe, Pyxis, Skyphos
- Ground truth classification for evaluation

### 3D API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /models3d/upload | Upload a 3D model |
| POST | /models3d/descriptors | Extract descriptors from OBJ |
| GET | /models3d/index | Get index statistics |
| GET | /models3d/list | List all indexed models (with pagination) |
| POST | /models3d/index/build | Build index from directory |
| DELETE | /models3d/index | Clear the 3D index |
| POST | /models3d/search | Search similar models |
| POST | /models3d/evaluate | Evaluate system performance |
| GET | /models3d/{id} | Get model information |
| GET | /models3d/{id}/file | Download OBJ file |

### 3D Visualization (Three.js)

The system includes an interactive WebGL-based 3D viewer powered by **Three.js**:

**Features:**
- **Interactive viewing**: Rotate, zoom, and pan 3D models with mouse controls
- **Auto-rotation**: Automatic model rotation for better visualization
- **Wireframe mode**: Toggle between solid and wireframe rendering
- **Color customization**: Choose from preset colors for the model
- **Fullscreen support**: Expand the viewer to fullscreen mode
- **Gallery preview**: Quick preview modal in the gallery without leaving the page
- **Search results preview**: Load 3D previews directly in search results

**Viewer Controls:**
- **Left click + drag**: Rotate the model
- **Right click + drag**: Pan the view
- **Scroll wheel**: Zoom in/out
- **Toggle buttons**: Wireframe, auto-rotate, color options

**Implementation:**
- Uses Three.js (v0.128.0) with OBJLoader for loading Wavefront OBJ files
- OrbitControls for camera interaction
- Responsive design that adapts to container size
- Custom `Model3DViewer` class in `static/js/viewer3d.js`

### 3D Evaluation Metrics

The 3D search system computes and displays evaluation metrics in real-time when searching:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **P@K** | Relevant in top-K / K | Precision of top results |
| **NDCG@K** | DCG@K / IDCG@K | Ranking quality |
| **AP** | Average Precision | Overall search quality |

**Evaluation in Search Results:**

When searching for similar 3D models, if the query model has a known category, the system automatically computes:
- **P@5, P@10**: Precision at top 5 and 10 results
- **Average Precision (AP)**: Mean precision across all relevant results
- **NDCG@10**: Normalized Discounted Cumulative Gain

These metrics appear in a dedicated panel above the search results.

**Batch Evaluation:**

For comprehensive system evaluation, use the **Évaluation** page (3D Models → Évaluation):
```python
# Via API
response = api_client.post('/models3d/evaluate', {
    'metric': 'cosine',
    'k_values': [5, 10, 20]
})
# Returns: mAP, per-category P@K, NDCG@K
```

### Usage (3D Module)

1. Navigate to the **Recherche 3D** menu in the web interface
2. **Galerie**: View all indexed 3D models
3. **Upload**: Add new OBJ models to the system
4. **Construire Index**: Batch index models from a directory
5. **Rechercher**: Search for similar models by uploading a query
6. **Évaluer**: Run evaluation on the indexed collection

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
| 3D Processing | NumPy, SciPy (mesh operations) |
| 3D Descriptors | Custom implementation (Spin Image, PFH, etc.) |

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

