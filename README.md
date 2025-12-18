# CBIR System - Object Detection on Large Collection of Images

A Content-Based Image Retrieval (CBIR) system with YOLOv8 object detection and advanced feature descriptors.

## Project Structure

```
├── Data/                      # ImageNet dataset (15 categories)
├── yolo_training/             # YOLO training scripts
│   ├── data.yaml              # Dataset configuration
│   ├── convert_annotations.py # VOC to YOLO converter
│   └── train.py               # Training script
├── flask_api/                 # Flask REST API (ML services)
│   ├── app.py                 # Main Flask app
│   ├── config.py              # Configuration
│   ├── services/              # ML services
│   │   ├── detection.py       # YOLOv8 detection
│   │   ├── descriptors.py     # Feature extraction
│   │   └── similarity.py      # Similarity search
│   └── resources/             # API endpoints
├── django_app/                # Django web application
│   ├── config/                # Django settings
│   ├── core/                  # Main app (models, views)
│   ├── templates/             # HTML templates
│   └── static/                # CSS, JS files
└── requirements.txt           # Global Python requirements
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset (YOLO Format)

```bash
cd yolo_training
python convert_annotations.py
```

### 4. Train YOLO Model (Optional)

```bash
python train.py
```

### 5. Start Flask API

```bash
cd flask_api
python app.py
```
Flask API runs at: http://localhost:5000

### 6. Setup Django App

```bash
cd django_app
python manage.py migrate
python manage.py createsuperuser  # Optional
python manage.py runserver
```
Django app runs at: http://localhost:8000

## Features

### Object Detection (YOLOv8n)
- 15 classes: pineapple, apple, bell_pepper, bicycle, broccoli, bus, car, cat, dog, elephant, horse, lemon, motorcycle, strawberry, tomato
- Real-time detection with bounding boxes
- Confidence scores

### Feature Descriptors

**Color:**
- HSV Histograms
- Dominant Colors (K-means)
- Color Moments

**Texture:**
- Tamura (coarseness, contrast, directionality)
- Gabor Filters (4 scales × 6 orientations)
- LBP (Local Binary Pattern)
- GLCM (Gray Level Co-occurrence Matrix)

**Shape:**
- Hu Moments (7 invariants)
- HOG (Histogram of Oriented Gradients)
- Contour Features

### Similarity Search
- Distance metrics: Cosine, Euclidean, Manhattan, Chi-Square, Histogram Intersection
- Query by image upload
- Filter by category

### Image Transformations
- Resize, Rotate, Flip
- Brightness/Contrast adjustment
- Filters: Grayscale, Blur, Sharpen, Edge Detection

## API Endpoints (Flask)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | POST | Detect objects in image |
| `/api/detect/batch` | POST | Batch detection |
| `/api/descriptors/extract` | POST | Extract features |
| `/api/search/similar` | POST | Find similar images |
| `/api/images/upload` | POST | Upload image |
| `/api/images/<id>/transform` | POST | Transform image |

## Tech Stack

- **Detection:** YOLOv8n (Ultralytics)
- **Backend API:** Flask + Flask-RESTful
- **Frontend:** Django 5.0 + Bootstrap 5
- **Database:** SQLite (dev) / PostgreSQL (prod)
- **Image Processing:** OpenCV, Pillow, scikit-image

## License

Academic Project © 2025
