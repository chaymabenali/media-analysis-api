from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import yt_dlp
import os
import logging
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ========== CONFIGURATION MONGODB ==========
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = os.environ.get('DB_NAME', 'n8n_project')
COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'media_files')

# Connexion MongoDB
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    logger.info(f"✅ Connecté à MongoDB: {DB_NAME}.{COLLECTION_NAME}")
except Exception as e:
    logger.error(f"❌ Erreur connexion MongoDB: {str(e)}")
    mongo_client = None
    db = None
    collection = None

# Charger les modèles au démarrage
logger.info("Chargement de YOLO11n...")
model = YOLO('yolo11n.pt')

logger.info("Chargement de EasyOCR...")
reader = easyocr.Reader(['en', 'fr'], gpu=False)

# Mots-clés pour détecter les pop-ups
POPUP_KEYWORDS = [
    'click here', 'buy now', 'subscribe', 'sign up', 'register',
    'download now', 'get started', 'free trial', 'limited offer',
    'act now', 'close', 'dismiss', 'skip ad', 'advertisement',
    'cliquez ici', 'acheter', 'abonnez-vous', 'inscrivez-vous',
    'télécharger', 'essai gratuit', 'offre limitée', 'publicité'
]

def detect_objects_yolo(image):
    """Détecte les objets dans une image avec YOLO11"""
    try:
        results = model(image)
        objects = {}
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                objects[cls_name] = objects.get(cls_name, 0) + 1
        
        return objects
    except Exception as e:
        logger.error(f"Erreur YOLO: {str(e)}")
        return {}

def extract_text_ocr(image):
    """Extrait le texte d'une image avec EasyOCR"""
    try:
        # Convertir en numpy array si nécessaire
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # OCR
        results = reader.readtext(image)
        text = ' '.join([item[1] for item in results])
        
        return text.strip()
    except Exception as e:
        logger.error(f"Erreur OCR: {str(e)}")
        return ""

def detect_popup(text):
    """Détecte si le texte contient des mots-clés de pop-up"""
    if not text:
        return False
    
    text_lower = text.lower()
    for keyword in POPUP_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def download_image(url):
    """Télécharge une image depuis une URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return np.array(image)
    except Exception as e:
        logger.error(f"Erreur téléchargement image: {str(e)}")
        raise

def download_video_frames(url, max_frames=5):
    """Télécharge et extrait des frames d'une vidéo"""
    try:
        # Options pour yt-dlp
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']
        
        # Télécharger la vidéo
        cap = cv2.VideoCapture(video_url)
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // max_frames, 1)
        
        while len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += step
        
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"Erreur téléchargement vidéo: {str(e)}")
        raise

def save_analysis_to_mongodb(url, media_type, yolo_objects, text_extracted, has_popup, frames_analyzed=None):
    """Sauvegarde les résultats d'analyse dans MongoDB"""
    if collection is None:
        logger.warning("MongoDB non connecté, résultats non sauvegardés")
        return None
    
    try:
        # Chercher si l'URL existe déjà
        existing = collection.find_one({"url": url})
        
        analysis_data = {
            "yolo_objects": yolo_objects,
            "text_extracted": text_extracted,
            "has_popup": has_popup,
            "analyzed": True,
            "analyzed_at": datetime.utcnow()
        }
        
        if frames_analyzed:
            analysis_data["frames_analyzed"] = frames_analyzed
        
        if existing:
            # Mettre à jour le document existant
            result = collection.update_one(
                {"_id": existing["_id"]},
                {"$set": analysis_data}
            )
            logger.info(f"✅ Document mis à jour: {existing['_id']}")
            return str(existing["_id"])
        else:
            # Créer un nouveau document
            new_doc = {
                "url": url,
                "name": url.split('/')[-1],
                "media_type": media_type,
                **analysis_data
            }
            result = collection.insert_one(new_doc)
            logger.info(f"✅ Nouveau document créé: {result.inserted_id}")
            return str(result.inserted_id)
            
    except Exception as e:
        logger.error(f"Erreur sauvegarde MongoDB: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé"""
    mongodb_status = collection is not None
    
    return jsonify({
        'status': 'healthy',
        'models': {
            'yolo': model is not None,
            'ocr': reader is not None
        },
        'mongodb': mongodb_status
    })

@app.route('/analyze/image', methods=['POST'])
def analyze_image():
    """Analyse une image"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL manquante'}), 400
        
        url = data['url']
        logger.info(f"Analyse de l'image: {url}")
        
        # Télécharger l'image
        image = download_image(url)
        
        # Détection d'objets
        objects = detect_objects_yolo(image)
        
        # Extraction de texte
        text = extract_text_ocr(image)
        
        # Détection de pop-up
        has_popup = detect_popup(text)
        
        # Sauvegarder dans MongoDB
        doc_id = save_analysis_to_mongodb(url, 'image', objects, text, has_popup)
        
        response = {
            'success': True,
            'media_type': 'image',
            'url': url,
            'yolo_objects': objects,
            'text_extracted': text,
            'has_popup': has_popup
        }
        
        if doc_id:
            response['mongodb_id'] = doc_id
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur analyse image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analyze/video', methods=['POST'])
def analyze_video():
    """Analyse une vidéo"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL manquante'}), 400
        
        url = data['url']
        max_frames = data.get('max_frames', 5)
        logger.info(f"Analyse de la vidéo: {url}")
        
        # Télécharger les frames
        frames = download_video_frames(url, max_frames)
        
        if not frames:
            return jsonify({'error': 'Aucune frame extraite'}), 500
        
        # Analyser chaque frame
        all_objects = {}
        all_texts = []
        
        for frame in frames:
            objects = detect_objects_yolo(frame)
            for obj, count in objects.items():
                all_objects[obj] = all_objects.get(obj, 0) + count
            
            text = extract_text_ocr(frame)
            if text:
                all_texts.append(text)
        
        # Consolider les résultats
        combined_text = ' '.join(all_texts)
        has_popup = detect_popup(combined_text)
        
        # Sauvegarder dans MongoDB
        doc_id = save_analysis_to_mongodb(
            url, 'video', all_objects, combined_text, has_popup, len(frames)
        )
        
        response = {
            'success': True,
            'media_type': 'video',
            'url': url,
            'frames_analyzed': len(frames),
            'yolo_objects': all_objects,
            'text_extracted': combined_text,
            'has_popup': has_popup
        }
        
        if doc_id:
            response['mongodb_id'] = doc_id
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur analyse vidéo: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """Analyse tous les fichiers non analysés dans MongoDB"""
    if collection is None:
        return jsonify({'error': 'MongoDB non connecté'}), 500
    
    try:
        # Trouver tous les documents non analysés
        unanalyzed = collection.find({"analyzed": False})
        
        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'details': []
        }
        
        for doc in unanalyzed:
            results['total'] += 1
            url = doc.get('url')
            
            if not url:
                results['failed'] += 1
                continue
            
            try:
                # Déterminer le type de média
                is_video = url.endswith(('.mp4', '.avi', '.mov')) or 'youtube.com' in url or 'youtu.be' in url
                
                if is_video:
                    # Analyser vidéo
                    frames = download_video_frames(url, 3)
                    all_objects = {}
                    all_texts = []
                    
                    for frame in frames:
                        objects = detect_objects_yolo(frame)
                        for obj, count in objects.items():
                            all_objects[obj] = all_objects.get(obj, 0) + count
                        text = extract_text_ocr(frame)
                        if text:
                            all_texts.append(text)
                    
                    combined_text = ' '.join(all_texts)
                    has_popup = detect_popup(combined_text)
                    save_analysis_to_mongodb(url, 'video', all_objects, combined_text, has_popup, len(frames))
                else:
                    # Analyser image
                    image = download_image(url)
                    objects = detect_objects_yolo(image)
                    text = extract_text_ocr(image)
                    has_popup = detect_popup(text)
                    save_analysis_to_mongodb(url, 'image', objects, text, has_popup)
                
                results['success'] += 1
                results['details'].append({'url': url, 'status': 'success'})
                
            except Exception as e:
                results['failed'] += 1
                results['details'].append({'url': url, 'status': 'failed', 'error': str(e)})
                logger.error(f"Erreur analyse {url}: {str(e)}")
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Erreur batch: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil"""
    return jsonify({
        'name': 'Media Analysis API',
        'version': '2.0.0',
        'mongodb': collection is not None,
        'endpoints': {
            'health': '/health',
            'analyze_image': '/analyze/image (POST)',
            'analyze_video': '/analyze/video (POST)',
            'analyze_batch': '/analyze/batch (POST) - Analyse tous les fichiers non analysés dans MongoDB'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
