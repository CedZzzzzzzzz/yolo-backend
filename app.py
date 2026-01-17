from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
from dotenv import load_dotenv 
import google.generativeai as genai
from datetime import datetime
from rag_engine import RAGEngine

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

# Initialize RAG Engine
print("\n🔧 Initializing RAG Engine...")
rag_engine = RAGEngine(documents_folder="rules_documents")

# Load YOLO Model
try:
    model = YOLO('best.pt')
    print("✅ Custom YOLO model loaded successfully")
except Exception as e:
    print(f"⚠️ Custom model not found: {e}, using fallback")
    try:
        import torch.serialization
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
        model = YOLO('yolov8n.pt')
        print("✅ Fallback YOLO model loaded successfully")
    except Exception as e2:
        print(f"❌ Could not load YOLO model: {e2}")
        # Downgrade torch as last resort
        print("💡 Try: pip install torch==2.1.0 torchvision==0.16.0")
        raise

CLASS_NAMES = model.names if hasattr(model, 'names') else {}
THREAT_CLASSES = ['knife', 'tank', 'rifle', 'pistol', 'blade', 'bomb', 'blood', 'weapon', 'gun', 'firearm', 'ammunition', 'bullets']

@app.route('/')
def home():
    return jsonify({
        "status": "Detective Investigation System Active (RAG Enabled)",
        "endpoints": {
            "/detect": "POST - Analyze crime scene",
            "/chat": "POST - Investigation assistant with RAG"
        },
        "rag_status": "Active" if rag_engine.vectorstore else "No documents loaded"
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

                # Check if this is a threat
                is_threat = any(threat.lower() in class_name.lower() for threat in THREAT_CLASSES)
                if is_threat:
                    threat_objects.append(detection_obj)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        primary_threat = class_name
        
        # Determine threat status
        threat_detected = len(threat_objects) > 0
        
        # Create detection summary for title
        detection_summary = create_detection_summary(all_detections, threat_objects)
        
        # Determine threat level
        if threat_detected:
            threat_level = "high" if max_confidence * 100 > 70 else "medium"
        else:
            threat_level = "low"
        
        # Main object and confidence
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
    
    # Count objects by type
    object_counts = {}
    for det in all_detections:
        obj_name = det['class_name']
        object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
    
    # If threats detected, prioritize them
    if threat_objects:
        threat_counts = {}
        for det in threat_objects:
            obj_name = det['class_name']
            threat_counts[obj_name] = threat_counts.get(obj_name, 0) + 1
        
        # Build threat summary
        threat_parts = []
        for obj, count in threat_counts.items():
            if count > 1:
                threat_parts.append(f"{count} {obj}s")
            else:
                threat_parts.append(f"{count} {obj}")
        
        return ", ".join(threat_parts) + " Detected"
    
    # No threats - general summary
    if len(object_counts) == 1:
        obj_name = list(object_counts.keys())[0]
        count = object_counts[obj_name]
        if count > 1:
            return f"{count} {obj_name}s Detected"
        return f"{obj_name} Detected"
    
    # Multiple object types
    return f"{len(all_detections)} Objects Detected"

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    """Detective AI investigation assistant with RAG"""
    try:
        data = request.get_json()
        user_question = data.get('question', '')
        detections = data.get('detections', [])
        
        # Get detection context
        det_list = [d.get('class_name', 'object') for d in detections]
        context_str = ", ".join(det_list) if det_list else "no objects"
        
        # Get relevant rules/regulations from RAG
        rag_context = ""
        if rag_engine.vectorstore:
            results = rag_engine.search(user_question, k=2)  
            
            if results:
                # Format context without page numbers - just cite source documents
                sources = set()
                context_parts = []
                for result in results:
                    source = os.path.basename(result['source'])
                    sources.add(source.replace('.pdf', '').replace('-', ' ').title())
                    context_parts.append(result['content'])
                
                rag_context = "\n\n".join(context_parts)
                source_citation = ", ".join(sources)
            else:
                rag_context = None
                source_citation = None
        
        # Build enhanced prompt
        if rag_context:
            prompt = f"""Act as a professional detective and analytical assistant specialized in evidence-based scenarios.

**Evidence Detected:** {context_str}

**Relevant Regulations:**
{rag_context}
[Source: {source_citation}]

**Question:** "{user_question}"

**Instructions:**
- Provide direct, concise answers for simple questions
- For regulation-related queries, cite sources as "According to [source name], ..."
- Never mention page numbers
- Base answers on the regulations provided above
- Keep responses professional and to-the-point
- Only elaborate when the question requires detailed analysis"""
        else:
            # No RAG context - use detective expertise
            prompt = f"""Act as a professional detective analyzing this evidence.

**Evidence Detected:** {context_str}

**Question:** "{user_question}"

Provide a direct, professional answer based on your analytical expertise. Keep it concise unless the question requires detailed investigation."""

        response = chat_model.generate_content(prompt)
        
        return jsonify({
            "reply": response.text,
            "rag_used": bool(rag_context)
        })

    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return jsonify({"reply": "⚠️ Investigation AI temporarily unavailable. Please try again."}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Detective Investigation System")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=True)