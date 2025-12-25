#!/usr/bin/env python3
"""
Test file service imports step by step.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing file service imports...")

try:
    import os
    print("✓ os imported")
except Exception as e:
    print(f"✗ os failed: {e}")

try:
    import shutil
    print("✓ shutil imported")
except Exception as e:
    print(f"✗ shutil failed: {e}")

try:
    import mimetypes
    print("✓ mimetypes imported")
except Exception as e:
    print(f"✗ mimetypes failed: {e}")

try:
    from pathlib import Path
    print("✓ pathlib imported")
except Exception as e:
    print(f"✗ pathlib failed: {e}")

try:
    from typing import Optional, Tuple, List
    print("✓ typing imported")
except Exception as e:
    print(f"✗ typing failed: {e}")

try:
    from fastapi import UploadFile, HTTPException
    print("✓ fastapi imported")
except Exception as e:
    print(f"✗ fastapi failed: {e}")

try:
    import librosa
    print("✓ librosa imported")
except Exception as e:
    print(f"✗ librosa failed: {e}")

try:
    import ffmpeg
    print("✓ ffmpeg imported")
except Exception as e:
    print(f"✗ ffmpeg failed: {e}")

try:
    from app.core.config import settings
    print("✓ settings imported")
except Exception as e:
    print(f"✗ settings failed: {e}")

try:
    from app.models.file import ProcessingStatus
    print("✓ ProcessingStatus imported")
except Exception as e:
    print(f"✗ ProcessingStatus failed: {e}")

try:
    from app.models.session import UserSession
    print("✓ UserSession imported")
except Exception as e:
    print(f"✗ UserSession failed: {e}")

try:
    from app.services.session_service import FileAccessService
    print("✓ FileAccessService imported")
except Exception as e:
    print(f"✗ FileAccessService failed: {e}")

print("\nNow trying to import the full FileService...")

try:
    from app.services.file_service import FileValidationService
    print("✓ FileValidationService imported successfully!")
except Exception as e:
    print(f"✗ FileValidationService failed: {e}")

print("Test completed!")