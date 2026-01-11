from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
from dotenv import load_dotenv 
import google.generativeai as genai
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://crime-detection-system-steel.vercel.app"}})

# Initialize Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
else: 
    genai.configure(api_key=GOOGLE_API_KEY)
    chat_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini AI initialized successfully")

# Load YOLO Model
try:
    model = YOLO('best.pt')
    print("✅ Custom YOLO model loaded successfully")
except Exception as e:
    print(f"⚠️ Custom model not found: {e}, using fallback")
    model = YOLO('yolov8n.pt')

CLASS_NAMES = model.names if hasattr(model, 'names') else {}
THREAT_CLASSES = ['gun', 'knife', 'weapon', 'rifle', 'pistol', 'blade', 'bomb', 'blood']

@app.route('/')
def home():
    return jsonify({
        "status": "Backend is running",
        "endpoints": {
            "/detect": "POST - Object detection",
            "/chat": "POST - AI chatbot"
        }
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Main object detection endpoint - processes image with YOLO"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        results = model(img, conf=0.10)
        
        all_detections = []
        threat_detected = False
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
                    "bbox": box.xywh[0].tolist() if len(box.xywh) > 0 else []
                }
                all_detections.append(detection_obj)

                if any(threat.lower() in class_name.lower() for threat in THREAT_CLASSES):
                    threat_detected = True
                    if confidence > max_confidence:
                        max_confidence = confidence
                        primary_threat = class_name
        
        if all_detections:
            main_object = primary_threat if threat_detected else all_detections[0]["class_name"]
            main_conf = round(max_confidence * 100, 2) if threat_detected else all_detections[0]["confidence"]
        else:
            main_object = "None"
            main_conf = 0

        response = {
            "detected": threat_detected,
            "objectType": main_object,
            "confidence": main_conf,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_objects": len(all_detections),
            "all_detections": all_detections,
            "threat_level": "high" if (threat_detected and main_conf > 70) else ("medium" if threat_detected else "low")
        }
        return jsonify(response)

    except Exception as e:
        print(f"❌ Detection error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    """AI chatbot endpoint - processes questions with Gemini"""
    try:
        data = request.get_json()
        user_question = data.get('question', '')
        detections = data.get('detections', [])
        
        det_list = [d.get('class_name', 'object') for d in detections]
        context_str = ", ".join(det_list) if det_list else "no objects"
        
        prompt = f"""You are a professional security AI assistant analyzing a crime detection system.

Scene analysis: {context_str} detected.

User question: "{user_question}"

Provide a clear, professional response that:
1. Directly answers the user's question
2. Assesses danger level if weapons/blood are present
3. Provides safety recommendations for threats
4. Keeps tone calm but authoritative
5. Uses concise, easy-to-understand language

Be helpful and reassuring while maintaining security awareness."""

        response = chat_model.generate_content(prompt)
        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return jsonify({"reply": "⚠️ Error connecting to AI. Please try again."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)