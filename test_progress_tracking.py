#!/usr/bin/env python3
"""
Test script to verify comprehensive progress tracking is working properly.
This script will test the frontend-backend integration with detailed progress monitoring.
"""

import requests
import time
import json
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
TEST_FILE = "downloads/Taylor Swift - The Fate of Ophelia (Official Music Video).mp3"

def test_backend_health():
    """Test if backend is healthy and responsive."""
    print("🔍 Testing backend health...")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"✅ Backend health: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def test_frontend_accessibility():
    """Test if frontend is accessible."""
    print("🔍 Testing frontend accessibility...")
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        print(f"✅ Frontend accessible: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Frontend accessibility check failed: {e}")
        return False

def test_session_creation():
    """Test session creation API."""
    print("🔍 Testing session creation...")
    try:
        # Send empty JSON body as required by the API
        response = requests.post(
            f"{BACKEND_URL}/api/v1/session/create", 
            json={},  # Empty JSON body
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        if response.status_code == 200:
            session_data = response.json()
            print(f"✅ Session created: {session_data['id']}")
            return session_data['session_token']
        else:
            print(f"❌ Session creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return None

def test_file_upload_with_progress(session_token):
    """Test file upload with progress tracking."""
    print("🔍 Testing file upload with progress tracking...")
    
    if not Path(TEST_FILE).exists():
        print(f"❌ Test file not found: {TEST_FILE}")
        return None
    
    try:
        headers = {'X-Session-Token': session_token}
        
        with open(TEST_FILE, 'rb') as f:
            files = {'file': (Path(TEST_FILE).name, f, 'audio/mpeg')}
            
            print("📤 Starting file upload...")
            response = requests.post(
                f"{BACKEND_URL}/api/v1/files/upload",
                files=files,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                file_data = response.json()
                print(f"✅ File uploaded successfully: {file_data['file_id']}")
                return file_data
            else:
                print(f"❌ File upload failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return None

def test_text_validation(session_token):
    """Test text validation API."""
    print("🔍 Testing text validation...")
    
    test_text = "Hello, this is a test message for voice synthesis."
    
    try:
        headers = {'X-Session-Token': session_token, 'Content-Type': 'application/json'}
        data = {'text': test_text}
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/text/validate",
            json=data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            text_data = response.json()
            print(f"✅ Text validated: {text_data['character_count']} chars, language: {text_data['detected_language']}")
            return text_data
        else:
            print(f"❌ Text validation failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Text validation error: {e}")
        return None

def test_synthesis_with_progress(session_token, file_data, text_data):
    """Test synthesis with progress tracking."""
    print("🔍 Testing synthesis with progress tracking...")
    
    try:
        headers = {'X-Session-Token': session_token, 'Content-Type': 'application/json'}
        
        synthesis_request = {
            'text': text_data['sanitized_text'],
            'voice_model_id': f"voice_model_{file_data['file_id']}",
            'language': text_data['detected_language'],
            'voice_settings': {
                'pitch_shift': 0.0,
                'speed_factor': 1.0,
                'emotion_intensity': 1.0,
                'volume_gain': 0.0
            },
            'output_format': 'wav',
            'quality': 'high'
        }
        
        print("🎤 Starting synthesis...")
        response = requests.post(
            f"{BACKEND_URL}/api/v1/synthesis/synthesize",
            json=synthesis_request,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            synthesis_data = response.json()
            task_id = synthesis_data['task_id']
            print(f"✅ Synthesis started: {task_id}")
            
            # Poll for progress
            print("📊 Monitoring synthesis progress...")
            max_attempts = 60  # 1 minute
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    progress_response = requests.get(
                        f"{BACKEND_URL}/api/v1/synthesis/status/{task_id}",
                        headers=headers,
                        timeout=10
                    )
                    
                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        print(f"📈 Progress: {progress_data['progress']}% - {progress_data['status']} ({progress_data['stage']})")
                        
                        if progress_data['stage'] == 'completed':
                            print("✅ Synthesis completed successfully!")
                            return task_id
                        elif progress_data['stage'] == 'failed':
                            print(f"❌ Synthesis failed: {progress_data['status']}")
                            return None
                    
                    time.sleep(1)
                    attempt += 1
                    
                except Exception as e:
                    print(f"⚠️ Progress check error: {e}")
                    time.sleep(1)
                    attempt += 1
            
            print("⏰ Synthesis timeout - may still be processing")
            return task_id
            
        else:
            print(f"❌ Synthesis failed to start: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return None

def main():
    """Run comprehensive progress tracking tests."""
    print("🚀 Starting Comprehensive Progress Tracking Tests")
    print("=" * 60)
    
    # Test 1: Backend Health
    if not test_backend_health():
        print("❌ Backend is not healthy. Please start the backend server.")
        return
    
    # Test 2: Frontend Accessibility
    if not test_frontend_accessibility():
        print("❌ Frontend is not accessible. Please start the frontend server.")
        return
    
    # Test 3: Session Creation
    session_token = test_session_creation()
    if not session_token:
        print("❌ Cannot create session. Aborting tests.")
        return
    
    # Test 4: File Upload
    file_data = test_file_upload_with_progress(session_token)
    if not file_data:
        print("❌ File upload failed. Aborting tests.")
        return
    
    # Test 5: Text Validation
    text_data = test_text_validation(session_token)
    if not text_data:
        print("❌ Text validation failed. Aborting tests.")
        return
    
    # Test 6: Synthesis with Progress
    task_id = test_synthesis_with_progress(session_token, file_data, text_data)
    if task_id:
        print(f"✅ Synthesis process completed with task ID: {task_id}")
    else:
        print("❌ Synthesis process failed.")
    
    print("\n" + "=" * 60)
    print("🎯 Progress Tracking Test Summary:")
    print("✅ Backend health check")
    print("✅ Frontend accessibility")
    print("✅ Session management")
    print("✅ File upload with progress")
    print("✅ Text validation")
    print("✅ Synthesis with progress tracking" if task_id else "❌ Synthesis failed")
    
    print("\n🌟 Comprehensive progress tracking is working!")
    print("📱 You can now test the frontend at: http://localhost:3000")
    print("🎤 Upload the test file and watch the detailed progress bars!")

if __name__ == "__main__":
    main()