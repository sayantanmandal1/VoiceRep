#!/usr/bin/env python3
"""
Simple test to verify core functionality without full app startup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing core imports...")
    
    try:
        from app.core.config import settings
        print("✓ Config imported successfully")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from app.core.database import engine, Base
        print("✓ Database imported successfully")
    except Exception as e:
        print(f"✗ Database import failed: {e}")
        return False
    
    try:
        from app.models.file import ReferenceAudio
        print("✓ Models imported successfully")
    except Exception as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    try:
        from app.schemas.file import FileUploadResponse
        print("✓ Schemas imported successfully")
    except Exception as e:
        print(f"✗ Schemas import failed: {e}")
        return False
    
    try:
        from app.services.file_service import FileService
        print("✓ Services imported successfully")
    except Exception as e:
        print(f"✗ Services import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database connection."""
    print("\nTesting database connection...")
    
    try:
        from app.core.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_api_routes():
    """Test API route definitions without starting server."""
    print("\nTesting API route definitions...")
    
    try:
        from app.api.v1.endpoints import files, text, voice, synthesis, session, performance
        print("✓ All endpoint modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Endpoint import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Voice Style Replication - Core Functionality Test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_database()
    all_passed &= test_api_routes()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All core functionality tests passed!")
        print("The application appears to be properly configured.")
    else:
        print("✗ Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)