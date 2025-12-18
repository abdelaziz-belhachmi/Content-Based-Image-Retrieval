# Cahier de Charge - Système de Recherche d'Images par le Contenu

## 1. Présentation du Projet

### 1.1 Titre
**Développement d'une Application Web pour l'Exploration d'une Collection d'Images basée sur la Détection d'Objets et l'Analyse de Contenu Visuel**

### 1.2 Contexte
Avec l'explosion du volume de données images, la recherche efficace de contenu visuel spécifique représente un défi majeur. Ce projet académique vise à développer un système intelligent de recherche d'images par le contenu (CBIR - Content-Based Image Retrieval) intégrant la détection d'objets via YOLO.

### 1.3 Objectifs
- Identifier et localiser précisément les objets dans les images
- Extraire les caractéristiques visuelles fondamentales (couleur, texture, forme)
- Constituer une base d'indexation robuste
- Permettre la recherche d'images similaires basée sur le contenu visuel

---

## 2. Architecture du Système

### 2.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION WEB                                 │
│                           (Django)                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Upload    │  │  Galerie    │  │  Recherche  │  │ Résultats   │    │
│  │   Images    │  │  Images     │  │  Similaire  │  │ Affichage   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP/REST
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API REST (Flask)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Détection  │  │ Extraction  │  │  Calcul     │  │  Recherche  │    │
│  │   YOLO      │  │ Descripteurs│  │ Similarité  │  │  Images     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BASE DE DONNÉES                                   │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │   Images & Métadonnées  │  │   Descripteurs & Index              │  │
│  │   (SQLite/PostgreSQL)   │  │   (Vecteurs de caractéristiques)    │  │
│  └─────────────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Stack Technologique

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Frontend Web | Django Templates + HTML/CSS/JS | Framework Python intégré |
| Backend Web | Django | Framework web Python robuste |
| API REST | Flask + Flask-RESTful | Microframework léger pour services ML |
| Détection d'Objets | YOLOv8n (Ultralytics) | Léger, rapide, 3.2M paramètres |
| Traitement d'Images | OpenCV, scikit-image, Pillow | Bibliothèques standard |
| Extraction Features | NumPy, SciPy | Calcul scientifique |
| Base de Données | SQLite (dev) / PostgreSQL (prod) | Simplicité et robustesse |

---

## 3. Base de Données d'Images

### 3.1 Source
**ImageNet** - Base de données d'images annotées à grande échelle basée sur WordNet.

### 3.2 Catégories Sélectionnées (15 classes)

| # | Catégorie | Synset ID | Type |
|---|-----------|-----------|------|
| 1 | Ananas (Pineapple) | n07753275 | Fruit |
| 2 | Apple | n07742313 | Fruit |
| 3 | Bell Pepper | n07720875 | Légume |
| 4 | Bicycle | n02834778 | Véhicule |
| 5 | Broccoli | n07714990 | Légume |
| 6 | Bus | n02924116 | Véhicule |
| 7 | Car | n02958343 | Véhicule |
| 8 | Cat | n02121620 | Animal |
| 9 | Dog | n02084071 | Animal |
| 10 | Elephant | n02504013 | Animal |
| 11 | Horse | n02374451 | Animal |
| 12 | Lemon | n07749582 | Fruit |
| 13 | Motorcycle | n03790512 | Véhicule |
| 14 | Strawberry | n07745940 | Fruit |
| 15 | Tomato | n07734017 | Fruit/Légume |

### 3.3 Structure des Données

```
Data/
├── [Catégorie]/
│   ├── Annotation/
│   │   └── [synset_id]/
│   │       └── [synset_id]_[image_id].xml  (Format Pascal VOC)
│   └── [synset_id]/
│       └── [synset_id]_[image_id].JPEG
```

### 3.4 Statistiques du Dataset (Après filtrage)

| Catégorie | Images avec Annotations |
|-----------|------------------------|
| Ananas | 414 |
| Apple | 505 |
| BellPepper | 438 |
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
| **Total** | **6,092** |

---

## 4. Modules Fonctionnels

### 4.1 Module de Détection d'Objets (YOLOv8n)

#### 4.1.1 Spécifications
- **Modèle**: YOLOv8n (nano) - 3.2 millions de paramètres
- **Entraînement**: Fine-tuning sur les 15 catégories sélectionnées
- **Format d'entrée**: Images JPEG/PNG
- **Format de sortie**: Bounding boxes + classes + scores de confiance

#### 4.1.2 Pipeline de Détection
```
Image → Prétraitement → YOLOv8n → Post-traitement (NMS) → Objets détectés
```

#### 4.1.3 Données de Sortie par Objet
```json
{
  "class_id": 0,
  "class_name": "apple",
  "confidence": 0.92,
  "bbox": {
    "x_min": 120,
    "y_min": 80,
    "x_max": 340,
    "y_max": 290
  }
}
```

### 4.2 Module d'Extraction de Caractéristiques

#### 4.2.1 Descripteurs de Couleur

| Descripteur | Description | Dimension |
|-------------|-------------|-----------|
| Histogramme RGB | Distribution des couleurs dans l'espace RGB | 256 × 3 = 768 |
| Histogramme HSV | Distribution dans l'espace HSV | 180 + 256 + 256 = 692 |
| Couleurs Dominantes | K-means clustering (k=5) sur les pixels | 5 × 3 = 15 + poids |
| Moments de Couleur | Moyenne, écart-type, skewness par canal | 3 × 3 = 9 |

#### 4.2.2 Descripteurs de Texture

| Descripteur | Description | Dimension |
|-------------|-------------|-----------|
| **Tamura** | Coarseness, Contrast, Directionality, Line-likeness, Regularity, Roughness | 6 |
| **Filtres de Gabor** | Réponses à différentes fréquences et orientations (4 échelles × 6 orientations) | 24 × 2 = 48 |
| LBP (Local Binary Pattern) | Histogramme des patterns locaux | 256 |
| GLCM | Gray-Level Co-occurrence Matrix features | 4 × 4 = 16 |

#### 4.2.3 Descripteurs de Forme

| Descripteur | Description | Dimension |
|-------------|-------------|-----------|
| **Moments de Hu** | 7 moments invariants (translation, rotation, échelle) | 7 |
| **HOG** (Histogram of Oriented Gradients) | Histogramme des orientations du gradient | Variable |
| Contour | Périmètre, aire, circularité, solidité | 4 |
| Descripteurs de Fourier | Coefficients de la transformée de Fourier du contour | Variable |

#### 4.2.4 Vecteur de Caractéristiques Final

```
Vecteur = [Couleur | Texture | Forme]
        = [Hist_HSV | Couleurs_Dom | Moments_Couleur | Tamura | Gabor | LBP | Hu | HOG]
```

### 4.3 Module de Recherche par Similarité

#### 4.3.1 Métriques de Distance

| Métrique | Formule | Usage |
|----------|---------|-------|
| Distance Euclidienne | $d(x,y) = \sqrt{\sum_{i}(x_i - y_i)^2}$ | Général |
| Distance Cosinus | $d(x,y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$ | Vecteurs normalisés |
| Distance de Manhattan | $d(x,y) = \sum_{i}|x_i - y_i|$ | Histogrammes |
| Chi-Square | $\chi^2 = \sum_{i}\frac{(x_i - y_i)^2}{x_i + y_i}$ | Histogrammes |
| Intersection d'Histogrammes | $I(H_1, H_2) = \sum_i \min(H_1(i), H_2(i))$ | Histogrammes |

#### 4.3.2 Processus de Recherche

```
1. Image Requête → Détection YOLO → Objets détectés
2. Sélection d'un objet par l'utilisateur
3. Extraction des descripteurs de l'objet sélectionné
4. Calcul des distances avec tous les objets indexés
5. Tri par similarité (distance croissante)
6. Retour des K images les plus similaires
```

---

## 5. API REST (Flask)

### 5.1 Endpoints

#### 5.1.1 Détection d'Objets

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/detect` | Détecter les objets dans une image |
| POST | `/api/detect/batch` | Détecter les objets dans plusieurs images |

#### 5.1.2 Extraction de Descripteurs

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/descriptors/extract` | Extraire les descripteurs d'une image |
| POST | `/api/descriptors/extract/object` | Extraire les descripteurs d'un objet (crop) |
| GET | `/api/descriptors/{image_id}` | Récupérer les descripteurs stockés |

#### 5.1.3 Recherche par Similarité

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/search/similar` | Rechercher des images similaires |
| POST | `/api/search/by-object` | Rechercher par objet spécifique |

#### 5.1.4 Gestion des Images

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/images/upload` | Uploader une ou plusieurs images |
| DELETE | `/api/images/{image_id}` | Supprimer une image |
| GET | `/api/images/{image_id}` | Récupérer une image |

#### 5.1.5 Transformations

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/transform/crop` | Recadrer une image |
| POST | `/api/transform/resize` | Redimensionner une image |
| POST | `/api/transform/rotate` | Pivoter une image |

### 5.2 Format des Réponses

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2025-12-18T10:30:00Z"
}
```

---

## 6. Application Web (Django)

### 6.1 Fonctionnalités

#### 6.1.1 Gestion des Images
- [ ] Upload d'images (simple et multiple)
- [ ] Téléchargement d'images
- [ ] Suppression d'images
- [ ] Galerie avec pagination
- [ ] Filtrage par catégorie

#### 6.1.2 Transformations d'Images
- [ ] Recadrage (Crop)
- [ ] Redimensionnement (Resize)
- [ ] Rotation
- [ ] Sauvegarde comme nouvelle image

#### 6.1.3 Détection et Analyse
- [ ] Détection des objets (visualisation des bounding boxes)
- [ ] Affichage des descripteurs calculés
- [ ] Visualisation des caractéristiques (histogrammes, etc.)

#### 6.1.4 Recherche par Similarité
- [ ] Sélection d'une image requête
- [ ] Sélection d'un objet dans l'image
- [ ] Affichage des résultats triés par pertinence
- [ ] Configuration des paramètres de recherche

### 6.2 Structure du Projet Django

```
django_app/
├── manage.py
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── core/
│   ├── models.py          # Modèles de données
│   ├── views.py           # Vues principales
│   ├── urls.py            # Routes
│   └── forms.py           # Formulaires
├── api_client/
│   └── flask_client.py    # Client pour l'API Flask
├── templates/
│   ├── base.html
│   ├── gallery.html
│   ├── upload.html
│   ├── image_detail.html
│   ├── search.html
│   └── results.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
└── media/
    └── uploads/
```

### 6.3 Modèles de Données

```python
class Image(models.Model):
    file = models.ImageField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    category = models.CharField(max_length=100, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    width = models.IntegerField()
    height = models.IntegerField()

class DetectedObject(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE)
    class_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    x_min = models.IntegerField()
    y_min = models.IntegerField()
    x_max = models.IntegerField()
    y_max = models.IntegerField()

class Descriptor(models.Model):
    detected_object = models.OneToOneField(DetectedObject, on_delete=models.CASCADE)
    color_histogram = models.BinaryField()      # Serialized numpy array
    dominant_colors = models.JSONField()
    tamura_features = models.JSONField()
    gabor_features = models.BinaryField()
    hu_moments = models.JSONField()
    # ... autres descripteurs
```

---

## 7. Entraînement du Modèle YOLOv8n

### 7.1 Préparation des Données

#### 7.1.1 Conversion du Format
- **Entrée**: Pascal VOC (XML)
- **Sortie**: YOLO format (TXT)

```
# Format YOLO (une ligne par objet)
<class_id> <x_center> <y_center> <width> <height>
# Valeurs normalisées [0, 1]
```

#### 7.1.2 Structure pour l'Entraînement

```
yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

#### 7.1.3 Fichier data.yaml

```yaml
path: ./yolo_dataset
train: train/images
val: val/images

nc: 15  # nombre de classes

names:
  0: pineapple
  1: apple
  2: bell_pepper
  3: bicycle
  4: broccoli
  5: bus
  6: car
  7: cat
  8: dog
  9: elephant
  10: horse
  11: lemon
  12: motorcycle
  13: strawberry
  14: tomato
```

### 7.2 Configuration de l'Entraînement

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Charger le modèle pré-entraîné

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    save=True,
    device='cuda'  # ou 'cpu'
)
```

---

## 8. Structure Finale du Projet

```
Object_Detection_On_Large_Collection_of_Images/
│
├── Data/                           # Dataset ImageNet (15 catégories)
│   ├── Ananas/
│   ├── Apple/
│   └── ...
│
├── yolo_training/                  # Entraînement YOLO
│   ├── dataset/                    # Dataset format YOLO
│   ├── models/                     # Modèles entraînés
│   ├── convert_annotations.py     # Conversion VOC → YOLO
│   └── train.py                    # Script d'entraînement
│
├── flask_api/                      # API REST Flask
│   ├── app.py                      # Point d'entrée
│   ├── config.py                   # Configuration
│   ├── requirements.txt
│   ├── services/
│   │   ├── detection.py            # Service détection YOLO
│   │   ├── descriptors.py          # Extraction descripteurs
│   │   └── similarity.py           # Recherche similarité
│   ├── resources/
│   │   ├── detection.py            # Endpoints détection
│   │   ├── descriptors.py          # Endpoints descripteurs
│   │   ├── images.py               # Endpoints images
│   │   └── search.py               # Endpoints recherche
│   └── utils/
│       ├── image_processing.py
│       └── feature_extraction.py
│
├── django_app/                     # Application Web Django
│   ├── manage.py
│   ├── config/
│   ├── core/
│   ├── templates/
│   ├── static/
│   └── media/
│
├── notebooks/                      # Jupyter notebooks (exploration)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_similarity_search.ipynb
│
├── docs/                           # Documentation
│   └── cahier_de_charge.md
│
├── filter_data.py                  # Script de nettoyage des données
├── requirements.txt                # Dépendances globales
└── README.md
```

---

## 9. Dépendances

### 9.1 Python (requirements.txt)

```
# Deep Learning & Computer Vision
ultralytics>=8.0.0          # YOLOv8
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
scikit-image>=0.21.0
Pillow>=10.0.0

# Feature Extraction
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Flask API
flask>=3.0.0
flask-restful>=0.3.10
flask-cors>=4.0.0

# Django Web App
django>=5.0
django-cors-headers>=4.3.0
requests>=2.31.0

# Database
psycopg2-binary>=2.9.9      # PostgreSQL (production)

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
matplotlib>=3.8.0
```

---

## 10. Livrables

1. **Code Source**
   - API Flask fonctionnelle
   - Application Django fonctionnelle
   - Scripts d'entraînement YOLO

2. **Modèle Entraîné**
   - Modèle YOLOv8n fine-tuné sur les 15 classes

3. **Documentation**
   - Cahier de charge (ce document)
   - README avec instructions d'installation
   - Documentation des API (Swagger/OpenAPI)

4. **Base de Données**
   - Images indexées avec descripteurs pré-calculés

---

## 11. Références

1. **YOLOv8**: https://docs.ultralytics.com/
2. **ImageNet**: https://www.image-net.org/
3. **Django**: https://docs.djangoproject.com/
4. **Flask-RESTful**: https://flask-restful.readthedocs.io/
5. **OpenCV**: https://docs.opencv.org/
6. **scikit-image**: https://scikit-image.org/docs/

---

*Document créé le: 18 Décembre 2025*
*Dernière mise à jour: 18 Décembre 2025*
