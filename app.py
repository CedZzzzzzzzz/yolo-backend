from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://crime-detection-system-steel.vercel.app"}})

# Load YOLO Model
try:
    model = YOLO('best.pt')
    print("✅ Custom YOLO model loaded successfully")
except Exception as e:
    print(f"⚠️ Custom model not found: {e}, using fallback")
    model = YOLO('yolov8n.pt')
    print("✅ Fallback YOLO model loaded successfully")

CLASS_NAMES = model.names if hasattr(model, 'names') else {}
THREAT_CLASSES = ['knife', 'tank', 'rifle', 'pistol', 'blade', 'bomb', 'blood', 'weapon', 'gun', 'firearm', 'ammunition', 'bullets']

@app.route('/')
def home():
    return jsonify({
        "status": "Crime Detection Service Active",
        "endpoints": {
            "/detect": "POST - Analyze crime scene"
        }
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Crime scene detection with bounding boxes"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        results = model(img, conf=0.10)
        
        all_detections = []
        threat_objects = []
        max_confidence = 0.0
        primary_threat = None

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                detection_obj = {
                    "class_name": class_name,
                    "confidence": round(confidence * 100, 2),
                    "bbox": box.xywh[0].tolist() if len(box.xywh) > 0 else [],
                    "xyxy": box.xyxy[0].tolist() if len(box.xyxy) > 0 else []
                }
                all_detections.append(detection_obj)

                is_threat = any(threat.lower() in class_name.lower() for threat in THREAT_CLASSES)
                if is_threat:
                    threat_objects.append(detection_obj)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        primary_threat = class_name
        
        threat_detected = len(threat_objects) > 0
        detection_summary = create_detection_summary(all_detections, threat_objects)
        
        if threat_detected:
            threat_level = "high" if max_confidence * 100 > 70 else "medium"
        else:
            threat_level = "low"
        
        if threat_objects:
            main_object = primary_threat
            main_conf = round(max_confidence * 100, 2)
        elif all_detections:
            main_object = all_detections[0]["class_name"]
            main_conf = all_detections[0]["confidence"]
        else:
            main_object = "None"
            main_conf = 0

        response = {
            "detected": threat_detected,
            "objectType": main_object,
            "confidence": main_conf,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_objects": len(all_detections),
            "threat_count": len(threat_objects),
            "all_detections": all_detections,
            "threat_objects": threat_objects,
            "threat_level": threat_level,
            "detection_summary": detection_summary
        }
        return jsonify(response)

    except Exception as e:
        print(f"❌ Detection error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def create_detection_summary(all_detections, threat_objects):
    """Create a human-readable summary of detections"""
    if not all_detections:
        return "No Objects Detected"
    
    object_counts = {}
    for det in all_detections:
        obj_name = det['class_name']
        object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
    
    if threat_objects:
        threat_counts = {}
        for det in threat_objects:
            obj_name = det['class_name']
            threat_counts[obj_name] = threat_counts.get(obj_name, 0) + 1
        
        threat_parts = []
        for obj, count in threat_counts.items():
            if count > 1:
                threat_parts.append(f"{count} {obj}s")
            else:
                threat_parts.append(f"{count} {obj}")
        
        return ", ".join(threat_parts) + " Detected"
    
    if len(object_counts) == 1:
        obj_name = list(object_counts.keys())[0]
        count = object_counts[obj_name]
        if count > 1:
            return f"{count} {obj_name}s Detected"
        return f"{obj_name} Detected"
    
    return f"{len(all_detections)} Objects Detected"

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Crime Detection Service")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=True)