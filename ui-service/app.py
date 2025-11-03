from flask import Flask, request, jsonify, render_template_string
import requests
import base64
from PIL import Image
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# AI service URL (will be set via environment variable in Docker)
AI_SERVICE_URL = os.getenv('AI_SERVICE_URL', 'http://localhost:5001')

# HTML template for the UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection Microservice</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
        }
        
        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        
        .upload-section, .results-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            border: 2px dashed #dee2e6;
        }
        
        .upload-form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .file-input {
            padding: 15px;
            border: 2px dashed #667eea;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-input:hover {
            background: #e9ecef;
            border-color: #764ba2;
        }
        
        .file-input input {
            display: none;
        }
        
        .file-label {
            font-size: 1.1em;
            color: #667eea;
            cursor: pointer;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            display: none;
        }
        
        .result.visible {
            display: block;
        }
        
        .detection-count {
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .detections {
            display: flex;
            flex-direction: column;
            gap: 15px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .detection-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .detection-class {
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }
        
        .detection-confidence {
            color: #28a745;
            font-weight: bold;
        }
        
        .detection-bbox {
            color: #6c757d;
            font-family: monospace;
            font-size: 0.9em;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.visible {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #f5c6cb;
            margin-top: 15px;
        }
        
        .service-status {
            text-align: center;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            font-weight: bold;
        }
        
        .status-online {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-offline {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Object Detection Microservice</h1>
            <p>Upload an image to detect objects using YOLO11x AI model</p>
        </div>
        
        <div class="content">
            <div class="upload-section">
                <h2>📤 Upload Image</h2>
                <div class="service-status" id="serviceStatus">
                    Checking service status...
                </div>
                <form class="upload-form" id="uploadForm" enctype="multipart/form-data">
                    <div class="file-input">
                        <label class="file-label">
                            📁 Choose Image File
                            <input type="file" name="image" accept="image/*" required>
                        </label>
                    </div>
                    <div id="fileName" style="text-align: center; color: #666; font-style: italic;"></div>
                    <button type="submit" class="btn" id="detectBtn">
                        🔍 Detect Objects
                    </button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing image with AI model...</p>
                </div>
            </div>
            
            <div class="results-section">
                <h2>📊 Detection Results</h2>
                <div class="result" id="result">
                    <div class="detection-count" id="detectionCount"></div>
                    <div class="detections" id="detections"></div>
                </div>
                <div id="errorMessage" class="error" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        // Check service status on page load
        async function checkServiceStatus() {
            const statusElement = document.getElementById('serviceStatus');
            try {
                const response = await fetch('/health');
                const data = await response.json();
                statusElement.textContent = '✅ Services are online and ready';
                statusElement.className = 'service-status status-online';
            } catch (error) {
                statusElement.textContent = '❌ Services are offline';
                statusElement.className = 'service-status status-offline';
            }
        }
        
        // File input handling
        const fileInput = document.querySelector('input[type="file"]');
        const fileName = document.getElementById('fileName');
        
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                fileName.textContent = `Selected: ${this.files[0].name}`;
            } else {
                fileName.textContent = '';
            }
        });
        
        // Form submission
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const file = fileInput.files[0];
            if (!file) {
                showError('Please select an image file');
                return;
            }
            
            // Validate file type
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file (JPEG, PNG, etc.)');
                return;
            }
            
            // Show loading state
            const detectBtn = document.getElementById('detectBtn');
            const loading = document.getElementById('loading');
            detectBtn.disabled = true;
            loading.classList.add('visible');
            hideError();
            hideResult();
            
            try {
                const formData = new FormData();
                formData.append('image', file);
                
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayResults(result);
                } else {
                    showError(result.error || 'Error processing image');
                }
            } catch (error) {
                console.error('Error:', error);
                showError('Network error: Unable to connect to AI service');
            } finally {
                detectBtn.disabled = false;
                loading.classList.remove('visible');
            }
        });
        
        function displayResults(result) {
            const resultElement = document.getElementById('result');
            const detectionCount = document.getElementById('detectionCount');
            const detectionsElement = document.getElementById('detections');
            
            detectionCount.textContent = `Objects Detected: ${result.count}`;
            
            if (result.detections && result.detections.length > 0) {
                let html = '';
                result.detections.forEach((detection, index) => {
                    html += `
                    <div class="detection-item">
                        <div class="detection-class">${index + 1}. ${detection.class}</div>
                        <div class="detection-confidence">Confidence: ${(detection.confidence * 100).toFixed(2)}%</div>
                        <div class="detection-bbox">
                            BBox: [${detection.bbox.x1}, ${detection.bbox.y1}, ${detection.bbox.x2}, ${detection.bbox.y2}]
                        </div>
                    </div>`;
                });
                detectionsElement.innerHTML = html;
            } else {
                detectionsElement.innerHTML = '<div class="detection-item">No objects detected</div>';
            }
            
            resultElement.classList.add('visible');
        }
        
        function showError(message) {
            const errorElement = document.getElementById('errorMessage');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
        
        function hideError() {
            const errorElement = document.getElementById('errorMessage');
            errorElement.style.display = 'none';
        }
        
        function hideResult() {
            const resultElement = document.getElementById('result');
            resultElement.classList.remove('visible');
        }
        
        // Initialize page
        checkServiceStatus();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate file type
        if not image_file.content_type.startswith('image/'):
            return jsonify({'error': 'File must be an image'}), 400
        
        logger.info(f"Processing image: {image_file.filename}")
        
        # Send image to AI service
        files = {'image': (image_file.filename, image_file, image_file.content_type)}
        
        try:
            response = requests.post(f'{AI_SERVICE_URL}/detect', files=files, timeout=30)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to AI service")
            return jsonify({'error': 'AI service is unavailable'}), 503
        except requests.exceptions.Timeout:
            logger.error("AI service request timeout")
            return jsonify({'error': 'AI service timeout'}), 504
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            logger.error(f"AI service error: {response.status_code} - {response.text}")
            return jsonify({'error': 'AI service error'}), 500
            
    except Exception as e:
        logger.error(f"UI service error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check if AI service is responsive
        ai_response = requests.get(f'{AI_SERVICE_URL}/health', timeout=5)
        ai_status = ai_response.json() if ai_response.status_code == 200 else {'status': 'unavailable'}
    except:
        ai_status = {'status': 'unavailable'}
    
    return jsonify({
        'status': 'UI service is running',
        'ai_service': ai_status
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
