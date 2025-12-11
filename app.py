from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
from PIL import Image
from datetime import datetime
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

# Load YOLO model
MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

# Define threat classes (adjust based on your model)
THREAT_CLASSES = ['gun', 'knife', 'weapon', 'blood']

def preprocess_image(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        image = preprocess_image(image_data)
        results = model(image, conf=0.25)
        
        detections = results[0].boxes
        detected_objects = []
        max_confidence = 0
        threat_detected = False
        detected_threat_type = 'none'
        
        if len(detections) > 0:
            for box in detections:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[cls_id]
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': round(confidence * 100, 2)
                })
                
                if class_name.lower() in [t.lower() for t in THREAT_CLASSES]:
                    threat_detected = True
                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_threat_type = class_name
        
        response = {
            'detected': threat_detected,
            'objectType': detected_threat_type if threat_detected else 'none',
            'confidence': round(max_confidence * 100, 2) if threat_detected else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'allDetections': detected_objects
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_classes': model.names if model else []
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Crime Detection API',
        'endpoints': {
            '/api/detect': 'POST - Detect objects in image',
            '/api/health': 'GET - Health check'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)