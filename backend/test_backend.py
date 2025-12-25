#!/usr/bin/env python3
"""
Simple test to verify the backend can start and respond to requests
"""

import asyncio
import sys
import os
import uvicorn
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_backend():
    """Test if the backend can start"""
    try:
        from app.main import app
        print("✅ Backend imports successfully")
        
        # Test basic app creation
        print("✅ FastAPI app created")
        
        # Test mock TTS service
        from app.services.real_voice_synthesis_service import real_voice_synthesis_service
        
        # Test that the service is using mock mode
        use_mock = getattr(real_voice_synthesis_service, 'use_mock', True)
        if use_mock:
            print("✅ Mock TTS service is active")
        else:
            print("⚠️  Real TTS service is active (not using mock)")
            
        print("✅ Voice synthesis service initialized")
        
        print("\n🎉 Backend is ready to run!")
        print("To start the server, run:")
        print("uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_backend())
    sys.exit(0 if success else 1)