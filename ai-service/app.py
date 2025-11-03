import numpy as np
from flask import Flask, request, jsonify
import logging
from ultralytics import YOLO
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
CONFIDENCE_THRESHOLD = 0.25
OUTPUT_DIR = "output"  # Directory to save results

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load YOLO model
try:
    model = YOLO('yolo11x.pt')
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

def save_detection_results(image_filename, results, output_image_path, json_output_path):
    """Save detection results as image with bounding boxes and JSON file"""
    
    # Process results for JSON output
    detections = []
    if len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    'bbox': {
                        'x1': float(box.xyxy[0][0].item()),
                        'y1': float(box.xyxy[0][1].item()),
                        'x2': float(box.xyxy[0][2].item()),
                        'y2': float(box.xyxy[0][3].item())
                    },
                    'confidence': float(box.conf[0].item()),
                    'class_id': int(box.cls[0].item()),
                    'class_name': result.names[int(box.cls[0].item())]
                }
                detections.append(detection)
    
    # Save the result image with bounding boxes
    if len(results) > 0:
        results[0].save(filename=output_image_path)
        logger.info(f"Output image saved: {output_image_path}")
    
    # Prepare JSON data
    json_data = {
        'image_filename': image_filename,
        'timestamp': datetime.now().isoformat(),
        'detections': detections,
        'detection_count': len(detections),
        'success': True
    }
    
    # Save JSON file
    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"JSON results saved: {json_output_path}")
    
    return json_data

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Endpoint for object detection with output saving"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    start_time = datetime.now()
    
    try:
        image_file = request.files['image']
        original_filename = image_file.filename
        logger.info(f"Received image file: {original_filename}")
        
        # Get confidence threshold from request (optional)
        conf_threshold = float(request.form.get('confidence', CONFIDENCE_THRESHOLD))
        
        # Generate unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = Path(original_filename).stem
        output_filename = f"{timestamp}_{base_filename}"
        
        # Create temporary file for processing
        file_ext = os.path.splitext(original_filename)[1].lower()
        if not file_ext:
            file_ext = '.jpg'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            image_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Run detection with confidence threshold
            results = model(temp_file_path, conf=conf_threshold)
            
            # Define output paths
            output_image_path = os.path.join(OUTPUT_DIR, f"{output_filename}_detected.jpg")
            json_output_path = os.path.join(OUTPUT_DIR, f"{output_filename}_results.json")
            
            # Save results
            json_data = save_detection_results(
                original_filename, 
                results, 
                output_image_path, 
                json_output_path
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add file paths to response
            response = {
                **json_data,
                'output_files': {
                    'image': output_image_path,
                    'json': json_output_path
                },
                'processing_time': processing_time,
                'confidence_threshold': conf_threshold
            }
            
            logger.info(f"Detection completed: {len(json_data['detections'])} objects found in {processing_time:.2f}s")
            return jsonify(response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/results', methods=['GET'])
def list_results():
    """Endpoint to list all saved results"""
    try:
        image_files = []
        json_files = []
        
        for file in Path(OUTPUT_DIR).iterdir():
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png'] and '_detected' in file.name:
                image_files.append({
                    'filename': file.name,
                    'path': str(file),
                    'size': file.stat().st_size
                })
            elif file.suffix.lower() == '.json' and '_results' in file.name:
                json_files.append({
                    'filename': file.name,
                    'path': str(file),
                    'size': file.stat().st_size
                })
        
        return jsonify({
            'image_files': sorted(image_files, key=lambda x: x['filename'], reverse=True),
            'json_files': sorted(json_files, key=lambda x: x['filename'], reverse=True),
            'total_results': len(image_files)
        })
    
    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    """Endpoint to get specific result JSON"""
    try:
        json_path = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(json_path):
            return jsonify({'error': 'Result file not found'}), 404
        
        with open(json_path, 'r') as f:
            result_data = json.load(f)
        
        return jsonify(result_data)
    
    except Exception as e:
        logger.error(f"Error reading result file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'service': 'YOLO Object Detection',
        'output_directory': OUTPUT_DIR,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'YOLO Object Detection Service with Output Saving',
        'endpoints': {
            'POST /detect': 'Upload an image for object detection (saves output image and JSON)',
            'GET /results': 'List all saved results',
            'GET /results/<filename>': 'Get specific result JSON',
            'GET /health': 'Service health check'
        },
        'output_directory': OUTPUT_DIR
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
