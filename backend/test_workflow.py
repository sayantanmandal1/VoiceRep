#!/usr/bin/env python3
"""
Test script to verify the complete voice replication workflow.
"""

import requests
import json
import time
import os

# Configuration
BASE_URL = "http://localhost:8001/api/v1"
TEST_FILE = "downloads/Taylor Swift - The Fate of Ophelia (Official Music Video).mp3"

def test_workflow():
    """Test the complete workflow: session -> upload -> synthesis."""
    
    print("🚀 Starting Voice Replication Workflow Test")
    
    # Step 1: Create session
    print("\n1️⃣ Creating session...")
    session_response = requests.post(f"{BASE_URL}/session/create", json={})
    if session_response.status_code != 200:
        print(f"❌ Session creation failed: {session_response.text}")
        return False
    
    session_data = session_response.json()
    session_token = session_data["session_token"]
    print(f"✅ Session created: {session_data['id']}")
    
    headers = {"X-Session-Token": session_token}
    
    # Step 2: Upload file
    print("\n2️⃣ Uploading audio file...")
    if not os.path.exists(TEST_FILE):
        print(f"❌ Test file not found: {TEST_FILE}")
        return False
    
    with open(TEST_FILE, 'rb') as f:
        files = {'file': f}
        upload_response = requests.post(f"{BASE_URL}/files/upload", files=files, headers=headers)
    
    if upload_response.status_code != 200:
        print(f"❌ File upload failed: {upload_response.text}")
        return False
    
    upload_data = upload_response.json()
    file_id = upload_data["id"]
    print(f"✅ File uploaded: {file_id}")
    print(f"   Filename: {upload_data['filename']}")
    print(f"   Size: {upload_data['file_size']} bytes")
    
    # Step 3: Validate text
    print("\n3️⃣ Validating text...")
    text_data = {"text": "Hello, this is a test of voice synthesis using the uploaded audio file."}
    text_response = requests.post(f"{BASE_URL}/text/validate", json=text_data, headers=headers)
    
    if text_response.status_code != 200:
        print(f"❌ Text validation failed: {text_response.text}")
        return False
    
    text_result = text_response.json()
    print(f"✅ Text validated: {text_result['character_count']} characters")
    print(f"   Language: {text_result['detected_language']}")
    
    # Step 4: Start synthesis
    print("\n4️⃣ Starting speech synthesis...")
    synthesis_data = {
        "text": text_result["sanitized_text"],
        "voice_model_id": file_id,  # Use file ID directly
        "language": text_result["detected_language"],
        "output_format": "wav"
    }
    
    synthesis_response = requests.post(f"{BASE_URL}/synthesis/synthesize", json=synthesis_data, headers=headers)
    
    if synthesis_response.status_code != 200:
        print(f"❌ Synthesis failed: {synthesis_response.text}")
        return False
    
    synthesis_result = synthesis_response.json()
    task_id = synthesis_result["task_id"]
    print(f"✅ Synthesis started: {task_id}")
    print(f"   Status: {synthesis_result['status']}")
    
    # Step 5: Poll for completion
    print("\n5️⃣ Waiting for synthesis completion...")
    max_attempts = 30
    for attempt in range(max_attempts):
        status_response = requests.get(f"{BASE_URL}/synthesis/status/{task_id}", headers=headers)
        
        if status_response.status_code != 200:
            print(f"❌ Status check failed: {status_response.text}")
            return False
        
        status_data = status_response.json()
        progress = status_data.get("progress", 0)
        stage = status_data.get("stage", "unknown")
        
        print(f"   Progress: {progress}% - Stage: {stage}")
        
        if stage == "completed":
            print("✅ Synthesis completed!")
            break
        elif stage == "failed":
            print(f"❌ Synthesis failed: {status_data.get('status', 'Unknown error')}")
            return False
        
        time.sleep(2)
    else:
        print("❌ Synthesis timeout")
        return False
    
    # Step 6: Get result
    print("\n6️⃣ Getting synthesis result...")
    result_response = requests.get(f"{BASE_URL}/synthesis/result/{task_id}", headers=headers)
    
    if result_response.status_code != 200:
        print(f"❌ Result retrieval failed: {result_response.text}")
        return False
    
    result_data = result_response.json()
    print(f"✅ Result retrieved:")
    print(f"   Output URL: {result_data.get('output_url', 'N/A')}")
    print(f"   Processing time: {result_data.get('processing_time', 'N/A')}s")
    
    print("\n🎉 Workflow test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_workflow()
    exit(0 if success else 1)