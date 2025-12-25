#!/usr/bin/env python3
"""
Test session creation directly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app.main import app

def test_session_creation():
    """Test session creation directly."""
    client = TestClient(app)
    
    print("Testing session creation...")
    
    try:
        # Test session creation
        response = client.post("/api/v1/session/create", json={"user_identifier": "test_user"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            session_data = response.json()
            print(f"Session created successfully: {session_data['id']}")
            return session_data
        else:
            print(f"Session creation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error testing session: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_session_creation()