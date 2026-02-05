# 🎯 Media Analysis API

API Flask pour analyser des images et vidéos avec **YOLO11** et **EasyOCR**.

## 🚀 Fonctionnalités

- ✅ Détection d'objets (80 classes COCO)
- ✅ Extraction de texte (OCR en français et anglais)
- ✅ Détection de pop-ups publicitaires
- ✅ Support images et vidéos

## 📦 Technologies

- **Flask** : Framework web
- **YOLO11n** : Détection d'objets
- **EasyOCR** : Extraction de texte
- **OpenCV** : Traitement d'images
- **yt-dlp** : Téléchargement vidéos

## 🔌 Endpoints

### GET /health
Vérifier l'état de l'API

### POST /analyze/image
Analyser une image

```json
{
  "url": "https://example.com/image.jpg"
}
```

### POST /analyze/video
Analyser une vidéo

```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "max_frames": 5
}
```

## 🚀 Déploiement sur Render

1. Fork ce repository
2. Créer un Web Service sur Render
3. Connecter le repository
4. Attendre le déploiement (10-15 min)

## 📝 Licence

MIT