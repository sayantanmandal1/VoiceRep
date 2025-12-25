#!/usr/bin/env python3
"""
Simple API testing script to verify all endpoints are functional.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app.main import app
import json

def test_all_apis():
    """Test all API endpoints to ensure they're functional."""
    client = TestClient(app)
    
    print("Testing Voice Style Replication APIs...")
    print("=" * 50)
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = client.get("/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✓ Root endpoint working")
    except Exception as e:
        print(f"   ✗ Root endpoint failed: {e}")
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    try:
        response = client.get("/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✓ Health endpoint working")
    except Exception as e:
        print(f"   ✗ Health endpoint failed: {e}")
    
    # Test session endpoints
    print("\n3. Testing session endpoints...")
    try:
        # Create session
        response = client.post("/api/v1/session/create")
        print(f"   Create session status: {response.status_code}")
        if response.status_code == 200:
            session_data = response.json()
            session_id = session_data.get('session_id')
            print(f"   Session ID: {session_id}")
            print("   ✓ Session creation working")
            
            # Get session info
            if session_id:
                response = client.get(f"/api/v1/session/{session_id}")
                print(f"   Get session status: {response.status_code}")
                if response.status_code == 200:
                    print("   ✓ Session retrieval working")
                else:
                    print(f"   ✗ Session retrieval failed: {response.status_code}")
        else:
            print(f"   ✗ Session creation failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Session endpoints failed: {e}")
    
    # Test file endpoints
    print("\n4. Testing file endpoints...")
    try:
        # Test file upload endpoint structure
        response = client.post("/api/v1/files/upload", 
                             files={"file": ("test.txt", b"test content", "text/plain")})
        print(f"   Upload status: {response.status_code}")
        # We expect this to fail validation but endpoint should exist
        if response.status_code in [400, 422]:  # Validation error is expected
            print("   ✓ File upload endpoint exists and validates")
        elif response.status_code == 200:
            print("   ✓ File upload endpoint working")
        else:
            print(f"   ? File upload endpoint response: {response.status_code}")
    except Exception as e:
        print(f"   ✗ File endpoints failed: {e}")
    
    # Test text endpoints
    print("\n5. Testing text endpoints...")
    try:
        response = client.post("/api/v1/text/validate", 
                             json={"text": "Hello world", "language": "english"})
        print(f"   Text validation status: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ Text validation working")
        else:
            print(f"   ? Text validation response: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Text endpoints failed: {e}")
    
    # Test voice endpoints
    print("\n6. Testing voice endpoints...")
    try:
        response = client.get("/api/v1/voice/models")
        print(f"   Voice models status: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ Voice models endpoint working")
        else:
            print(f"   ? Voice models response: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Voice endpoints failed: {e}")
    
    # Test synthesis endpoints
    print("\n7. Testing synthesis endpoints...")
    try:
        response = client.get("/api/v1/synthesis/tasks")
        print(f"   Synthesis tasks status: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ Synthesis tasks endpoint working")
        else:
            print(f"   ? Synthesis tasks response: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Synthesis endpoints failed: {e}")
    
    # Test performance endpoints
    print("\n8. Testing performance endpoints...")
    try:
        response = client.get("/api/v1/performance/metrics")
        print(f"   Performance metrics status: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ Performance metrics working")
        else:
            print(f"   ? Performance metrics response: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Performance endpoints failed: {e}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")
    print("\nNote: Some endpoints may return validation errors when called without")
    print("proper parameters, but this confirms they exist and are accessible.")

if __name__ == "__main__":
    test_all_apis()