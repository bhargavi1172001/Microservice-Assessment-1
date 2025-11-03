#!/usr/bin/env python3
"""
Test script for Object Detection Microservice API
"""

import requests
import base64
from PIL import Image
import io
import json

def test_services():
    print("🧪 Testing Object Detection Microservice API...\n")
    
    # Test AI Service Health
    print("1. Testing AI Service Health...")
    try:
        response = requests.get("http://localhost:5001/health", timeout=10)
        print(f"   ✅ AI Service: {response.json()}")
    except Exception as e:
        print(f"   ❌ AI Service Health Check Failed: {e}")
        return
    
    # Test UI Service Health
    print("2. Testing UI Service Health...")
    try:
        response = requests.get("http://localhost:5000/health", timeout=10)
        print(f"   ✅ UI Service: {response.json()}")
    except Exception as e:
        print(f"   ❌ UI Service Health Check Failed: {e}")
        return
    
    # Test Model Info
    print("3. Testing Model Info...")
    try:
        response = requests.get("http://localhost:5001/model/info", timeout=10)
        print(f"   ✅ Model Info: {len(response.json().get('classes', []))} classes available")
    except Exception as e:
        print(f"   ❌ Model Info Failed: {e}")
    
    # Create a simple test image (red square)
    print("4. Creating test image...")
    try:
        # Create a simple colored image for testing
        img = Image.new('RGB', (640, 480), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Test detection with file upload
        print("5. Testing object detection with file upload...")
        files = {'image': ('test_image.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post("http://localhost:5001/detect", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Detection successful: {result['count']} objects detected")
            if result['detections']:
                for i, detection in enumerate(result['detections'][:3]):  # Show first 3
                    print(f"      {i+1}. {detection['class']} ({detection['confidence']:.2%})")
        else:
            print(f"   ❌ Detection failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"   ❌ Test image detection failed: {e}")
    
    print("\n🎉 API testing completed!")
    print("\n🌐 You can now access the web interface at: http://localhost:5000")

if __name__ == "__main__":
    test_services()
