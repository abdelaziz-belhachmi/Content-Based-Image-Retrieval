# Cahier de Charge - Systeme de Recherche d'Images par le Contenu

## 1. Presentation du Projet

### 1.1 Titre
Developpement d'une Application Web pour l'Exploration d'une Collection d'Images basee sur la Detection d'Objets et l'Analyse de Contenu Visuel

### 1.2 Contexte
Avec l'explosion du volume de donnees images, la recherche efficace de contenu visuel specifique represente un defi majeur. Ce projet academique vise a developper un systeme intelligent de recherche d'images par le contenu (CBIR - Content-Based Image Retrieval) integrant la detection d'objets via YOLO.

### 1.3 Objectifs
- Identifier et localiser precisement les objets dans les images
- Extraire les caracteristiques visuelles fondamentales (couleur, texture, forme)
- Constituer une base d'indexation robuste et persistante
- Permettre la recherche d'images similaires basee sur le contenu visuel des objets detectes

---

## 2. Architecture du Systeme

### 2.1 Vue d'Ensemble

```
+-------------------------------------------------------------------------+
|                          APPLICATION WEB                                 |
|                           (Django)                                       |
|  +-------------+  +-------------+  +-------------+  +-------------+     |
|  |   Upload    |  |  Galerie    |  |  Recherche  |  | Resultats   |     |
|  |   Images    |  |  Images     |  |  Similaire  |  | Affichage   |     |
|  +-------------+  +-------------+  +-------------+  +-------------+     |
+------------------------------------+------------------------------------+
                                     | HTTP/REST
                                     v
+-------------------------------------------------------------------------+
|                          API REST (Flask)                                |
|  +-------------+  +-------------+  +-------------+  +-------------+     |
|  |  Detection  |  | Extraction  |  |   Index     |  |  Recherche  |     |
|  |   YOLO      |  | Descripteurs|  | Persistant  |  |  Objets     |     |
|  +-------------+  +-------------+  +-------------+  +-------------+     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                        BASE DE DONNEES                                   |
|  +---------------------------+  +-----------------------------------+   |
|  |   Images & Metadonnees    |  |   Descripteurs & Index Objets    |   |
|  |   (SQLite/PostgreSQL)     |  |   (JSON persistant)              |   |
|  +---------------------------+  +-----------------------------------+   |
+-------------------------------------------------------------------------+
```

### 2.2 Stack Technologique

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Frontend Web | Django Templates + Bootstrap 5 | Framework Python integre |
| Backend Web | Django 5.0 | Framework web Python robuste |
| API REST | Flask + Flask-RESTful | Microframework leger pour services ML |
| Detection d'Objets | YOLOv8n (Ultralytics) | Leger, rapide, 3.2M parametres |
| Traitement d'Images | OpenCV, scikit-image, Pillow | Bibliotheques standard |
| Extraction Features | NumPy, SciPy, scikit-learn | Calcul scientifique |
| Base de Donnees | SQLite (dev) / PostgreSQL (prod) | Simplicite et robustesse |
| Persistance Index | JSON | Lisibilite et compatibilite |

---

## 3. Base de Donnees d'Images

### 3.1 Source
ImageNet - Base de donnees d'images annotees a grande echelle basee sur WordNet.

### 3.2 Categories Selectionnees (15 classes)

| # | Categorie | Synset ID | Type |
|---|-----------|-----------|------|
| 1 | Ananas (Pineapple) | n07753275 | Fruit |
| 2 | Apple | n07742313 | Fruit |
| 3 | Bell Pepper | n07720875 | Legume |
| 4 | Bicycle | n02834778 | Vehicule |
| 5 | Broccoli | n07714990 | Legume |
| 6 | Bus | n02924116 | Vehicule |
| 7 | Car | n02958343 | Vehicule |
| 8 | Cat | n02121620 | Animal |
| 9 | Dog | n02084071 | Animal |
| 10 | Elephant | n02504013 | Animal |
| 11 | Horse | n02374451 | Animal |
| 12 | Lemon | n07749582 | Fruit |
| 13 | Motorcycle | n03790512 | Vehicule |
| 14 | Strawberry | n07745940 | Fruit |
| 15 | Tomato | n07734017 | Fruit/Legume |

### 3.3 Structure des Donnees

```
Data/
+-- [Categorie]/
    +-- Annotation/
    |   +-- [synset_id]/
    |       +-- [synset_id]_[image_id].xml  (Format Pascal VOC)
    +-- [synset_id]/
        +-- [synset_id]_[image_id].JPEG
```

### 3.4 Statistiques du Dataset

| Categorie | Images avec Annotations |
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
| Total | 6,092 |

---

## 4. Modules Fonctionnels

### 4.1 Module de Detection d'Objets (YOLOv8n)

#### 4.1.1 Specifications
- Modele: YOLOv8n (nano) - 3.2 millions de parametres
- Entrainement: Fine-tuning sur les 15 categories selectionnees
- Format d'entree: Images JPEG/PNG
- Format de sortie: Bounding boxes + classes + scores de confiance

#### 4.1.2 Pipeline de Detection
```
Image -> Pretraitement -> YOLOv8n -> Post-traitement (NMS) -> Objets detectes
```

#### 4.1.3 Donnees de Sortie par Objet
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

### 4.2 Module d'Extraction de Caracteristiques

#### 4.2.1 Descripteurs de Couleur

| Descripteur | Description | Dimension |
|-------------|-------------|-----------|
| Histogramme HSV | Distribution dans l'espace HSV | Variable |
| Couleurs Dominantes | K-means clustering (k=5) sur les pixels | 5 x 3 = 15 + poids |
| Moments de Couleur | Moyenne, ecart-type, skewness par canal | 3 x 3 = 9 |

#### 4.2.2 Descripteurs de Texture

| Descripteur | Description | Dimension |
|-------------|-------------|-----------|
| Tamura | Coarseness, Contrast, Directionality | 6 |
| Filtres de Gabor | Reponses a differentes frequences et orientations (4 echelles x 6 orientations) | 48 |
| LBP | Local Binary Pattern - histogramme des patterns locaux | 256 |
| GLCM | Gray-Level Co-occurrence Matrix features | 16 |

#### 4.2.3 Descripteurs de Forme

| Descripteur | Description | Dimension |
|-------------|-------------|-----------|
| Moments de Hu | 7 moments invariants (translation, rotation, echelle) | 7 |
| HOG | Histogram of Oriented Gradients | Variable |
| Contour | Perimetre, aire, circularite, solidite | 4 |

#### 4.2.4 Vecteur de Caracteristiques Final

Le vecteur combine tous les descripteurs pour former une representation complete de chaque objet detecte:

```
Vecteur = [Couleur | Texture | Forme]
        = [Hist_HSV | Couleurs_Dom | Moments_Couleur | Tamura | Gabor | LBP | GLCM | Hu | Contour]
```

### 4.3 Module de Recherche par Similarite

#### 4.3.1 Metriques de Distance

| Metrique | Usage |
|----------|-------|
| Distance Cosinus | Vecteurs normalises (par defaut) |
| Distance Euclidienne | General |
| Distance de Manhattan | Histogrammes |
| Chi-Square | Histogrammes |
| Intersection d'Histogrammes | Histogrammes |

#### 4.3.2 Processus de Recherche Base sur les Objets

```
1. Image Requete -> Detection YOLO -> Objets detectes
2. Extraction des descripteurs pour chaque objet detecte
3. Recherche dans l'index: filtrage par classe d'objet
4. Calcul des distances avec les objets de meme classe
5. Agregation des scores par image
6. Retour des K images les plus similaires
```

#### 4.3.3 Caracteristiques Cles

- Filtrage par classe: Une image de cheval ne retourne que des images contenant des chevaux
- Index persistant: L'index est sauvegarde en JSON et survit aux redemarrages
- Indexation automatique: Chaque image uploadee est automatiquement indexee

---

## 5. API REST (Flask)

### 5.1 Endpoints de Detection

| Methode | Endpoint | Description |
|---------|----------|-------------|
| POST | /api/detect | Detecter les objets dans une image |
| POST | /api/detect/batch | Detection par lot |

### 5.2 Endpoints d'Extraction de Descripteurs

| Methode | Endpoint | Description |
|---------|----------|-------------|
| POST | /api/descriptors/extract | Extraire les descripteurs d'une image |
| POST | /api/descriptors/extract/object | Extraire les descripteurs d'un objet (crop) |

### 5.3 Endpoints de Gestion de l'Index

| Methode | Endpoint | Description |
|---------|----------|-------------|
| POST | /api/index/add | Ajouter une image a l'index |
| POST | /api/index/build | Construire l'index a partir de plusieurs images |
| GET | /api/index/stats | Statistiques de l'index |
| DELETE | /api/index/clear | Vider l'index |

### 5.4 Endpoints de Recherche

| Methode | Endpoint | Description |
|---------|----------|-------------|
| POST | /api/search/by-object | Recherche par objets detectes (principal) |
| POST | /api/search/similar | Recherche legacy par similarite |

### 5.5 Endpoints de Gestion des Images

| Methode | Endpoint | Description |
|---------|----------|-------------|
| POST | /api/images/upload | Uploader une image |
| GET | /api/images/{id} | Recuperer une image |
| DELETE | /api/images/{id} | Supprimer une image |

### 5.6 Endpoints de Transformations

| Methode | Endpoint | Description |
|---------|----------|-------------|
| POST | /api/transform/crop | Recadrer une image |
| POST | /api/transform/resize | Redimensionner une image |
| POST | /api/transform/rotate | Pivoter une image |
| POST | /api/transform/flip | Retourner une image |

### 5.7 Format des Reponses

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2025-01-01T10:30:00Z"
}
```

---

## 6. Application Web (Django)

### 6.1 Fonctionnalites Implementees

#### 6.1.1 Gestion des Images
- [x] Upload d'images (simple et multiple)
- [x] Telechargement d'images
- [x] Suppression d'images (simple et par lot)
- [x] Galerie avec pagination
- [x] Filtrage par categorie

#### 6.1.2 Transformations d'Images
- [x] Recadrage (Crop)
- [x] Redimensionnement (Resize)
- [x] Rotation
- [x] Retournement (Flip)

#### 6.1.3 Detection et Analyse
- [x] Detection automatique des objets a l'upload
- [x] Visualisation des bounding boxes
- [x] Affichage des descripteurs extraits
- [x] Indexation automatique pour la recherche

#### 6.1.4 Recherche par Similarite
- [x] Recherche par image de la galerie
- [x] Upload d'une image requete externe
- [x] Configuration des parametres (metrique, nombre de resultats)
- [x] Affichage des resultats avec scores de similarite

### 6.2 Structure du Projet Django

```
django_app/
+-- manage.py
+-- config/
|   +-- settings.py
|   +-- urls.py
|   +-- wsgi.py
+-- core/
|   +-- models.py          # Image, DetectedObject, Descriptor, SearchHistory
|   +-- views.py           # Vues principales
|   +-- api_client.py      # Client pour l'API Flask
|   +-- forms.py           # Formulaires
|   +-- urls.py            # Routes
+-- templates/
|   +-- core/
|       +-- base.html
|       +-- home.html
|       +-- gallery.html
|       +-- upload.html
|       +-- image_detail.html
|       +-- search.html
|       +-- object_search_results.html
|       +-- transform.html
+-- static/
|   +-- css/
|   +-- js/
+-- media/
    +-- uploads/
```

### 6.3 Modeles de Donnees

```python
class Image(models.Model):
    file = models.ImageField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    category = models.CharField(max_length=100, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    width = models.IntegerField()
    height = models.IntegerField()
    file_size = models.IntegerField()
    features_extracted = models.BooleanField(default=False)

class DetectedObject(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE)
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    x_min = models.IntegerField()
    y_min = models.IntegerField()
    x_max = models.IntegerField()
    y_max = models.IntegerField()

class Descriptor(models.Model):
    detected_object = models.OneToOneField(DetectedObject, on_delete=models.CASCADE)
    dominant_colors = models.JSONField(null=True)
    color_moments = models.JSONField(null=True)
    tamura_features = models.JSONField(null=True)
    glcm_features = models.JSONField(null=True)
    hu_moments = models.JSONField(null=True)
    contour_features = models.JSONField(null=True)

class SearchHistory(models.Model):
    query_image = models.ForeignKey(Image, on_delete=models.SET_NULL, null=True)
    metric_used = models.CharField(max_length=50)
    num_results = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
```

---

## 7. Entrainement du Modele YOLOv8n

### 7.1 Preparation des Donnees

#### 7.1.1 Conversion du Format
- Entree: Pascal VOC (XML)
- Sortie: YOLO format (TXT)

```
# Format YOLO (une ligne par objet)
<class_id> <x_center> <y_center> <width> <height>
# Valeurs normalisees [0, 1]
```

#### 7.1.2 Structure pour l'Entrainement

```
yolo_training/
+-- dataset/
|   +-- train/
|   |   +-- images/
|   |   +-- labels/
|   +-- val/
|       +-- images/
|       +-- labels/
+-- data.yaml
+-- convert_annotations.py
+-- train.py
+-- models/
    +-- yolov8n_finetuned_best.pt
    +-- yolov8n_finetuned_last.pt
```

#### 7.1.3 Fichier data.yaml

```yaml
path: ./dataset
train: train/images
val: val/images

nc: 15

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

### 7.2 Configuration de l'Entrainement

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    save=True,
    device='cuda'
)
```

---

## 8. Structure Finale du Projet

```
Content-Based-Image-Retrieval/
|
+-- Data/                           # Dataset ImageNet (15 categories)
|   +-- Ananas/
|   +-- Apple/
|   +-- ...
|
+-- yolo_training/                  # Entrainement YOLO
|   +-- dataset/                    # Dataset format YOLO
|   +-- models/                     # Modeles entraines
|   +-- convert_annotations.py
|   +-- train.py
|   +-- data.yaml
|
+-- flask_api/                      # API REST Flask
|   +-- app.py
|   +-- config.py
|   +-- requirements.txt
|   +-- services/
|   |   +-- detection.py
|   |   +-- descriptors.py
|   |   +-- similarity.py
|   |   +-- similarity_objects.py
|   |   +-- object_index_persistence.py
|   +-- resources/
|       +-- detection.py
|       +-- descriptors.py
|       +-- images.py
|       +-- search.py
|       +-- search_objects.py
|
+-- django_app/                     # Application Web Django
|   +-- manage.py
|   +-- config/
|   +-- core/
|   +-- templates/
|   +-- static/
|   +-- media/
|
+-- filter_data.py
+-- requirements.txt
+-- cahier_de_charge.md
+-- README.md
```

---

## 9. Dependances

### 9.1 Python (requirements.txt)

```
# Deep Learning & Computer Vision
ultralytics>=8.0.0
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
requests>=2.31.0

# Utilities
tqdm>=4.66.0
matplotlib>=3.8.0
```

---

## 10. Livrables

1. Code Source
   - API Flask fonctionnelle avec index persistant
   - Application Django fonctionnelle
   - Scripts d'entrainement YOLO
   - Scripts de conversion et preparation des donnees

2. Modele Entraine
   - Modele YOLOv8n fine-tune sur les 15 classes (yolov8n_finetuned_best.pt)

3. Documentation
   - Cahier de charge (ce document)
   - README avec instructions d'installation et d'utilisation

4. Base de Donnees
   - Schema SQLite avec images, objets detectes et descripteurs
   - Index JSON persistant pour la recherche rapide

---

## 11. References

1. YOLOv8: https://docs.ultralytics.com/
2. ImageNet: https://www.image-net.org/
3. Django: https://docs.djangoproject.com/
4. Flask-RESTful: https://flask-restful.readthedocs.io/
5. OpenCV: https://docs.opencv.org/
6. scikit-image: https://scikit-image.org/docs/

---

Document cree le: 18 Decembre 2025
Derniere mise a jour: 1 Janvier 2026
