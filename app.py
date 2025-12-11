from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image  # <--- NEW IMPORT
import io

app = Flask(__name__)
CORS(app)

# Load the model
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback if best.pt is missing
    model = YOLO('yolov8n.pt')

@app.route('/')
def home():
    return "YOLO Backend is Active!"

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # 🔧 FIX: Convert raw file to PIL Image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Run detection
        results = model(img)
        
        # Process results
        data = []
        for r in results:
            for box in r.boxes:
                # Get class name safely
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                data.append({
                    "objectType": class_name,
                    "confidence": float(box.conf[0]),
                    "bbox": box.xywh.tolist()
                })
        
        # Return result
        if len(data) > 0:
            return jsonify(data[0]) # Return first detection
        else:
            return jsonify({"detected": False, "message": "Nothing detected"})

    except Exception as e:
        print(f"ERROR: {str(e)}") # Print error to Render logs
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)