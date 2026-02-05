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

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'yolo': model is not None,
            'ocr': reader is not None
        }
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
        
        return jsonify({
            'success': True,
            'media_type': 'image',
            'url': url,
            'yolo_objects': objects,
            'text_extracted': text,
            'has_popup': has_popup
        })
    
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
        
        return jsonify({
            'success': True,
            'media_type': 'video',
            'url': url,
            'frames_analyzed': len(frames),
            'yolo_objects': all_objects,
            'text_extracted': combined_text,
            'has_popup': has_popup
        })
    
    except Exception as e:
        logger.error(f"Erreur analyse vidéo: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil"""
    return jsonify({
        'name': 'Media Analysis API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'analyze_image': '/analyze/image (POST)',
            'analyze_video': '/analyze/video (POST)'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)