from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load Model
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Custom model not found: {e}, using fallback.")
    model = YOLO('yolov8n.pt')

@app.route('/')
def home():
    return "Backend is Active!"

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        results = model(img, conf = 0.10)
        
        detection = None
        for r in results:
            for box in r.boxes:
                detection = {
                    "detected": True,
                    "objectType": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xywh.tolist()
                }
                break 
        
        if detection:
            return jsonify(detection)
        else:
            return jsonify({"detected": False, "message": "Nothing found"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)