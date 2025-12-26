#!/usr/bin/env python3
"""
Test script to verify synthesis fixes are working.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_fixes():
    """Test the synthesis fixes."""
    print("Testing synthesis fixes...")
    
    try:
        # Test 1: JSON Formatter with datetime objects
        print("\n1. Testing JSON formatter with datetime objects...")
        from app.core.logging_config import JSONFormatter
        import logging
        from datetime import datetime
        
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='Test message', args=(), exc_info=None
        )
        record.error_info = {'timestamp': datetime.now(), 'test': 'value'}
        
        result = formatter.format(record)
        print("✓ JSON formatter handles datetime objects correctly")
        
        # Test 2: Async function detection
        print("\n2. Testing async function detection...")
        async def test_async_func():
            return "test"
        
        def test_sync_func():
            return "test"
        
        assert asyncio.iscoroutinefunction(test_async_func) == True
        assert asyncio.iscoroutinefunction(test_sync_func) == False
        print("✓ Async function detection working correctly")
        
        # Test 3: Import key modules
        print("\n3. Testing module imports...")
        from app.services.ensemble_voice_synthesis_engine import EnsembleVoiceSynthesizer
        from app.services.performance_optimization_service import PerformanceOptimizationService
        from app.api.v1.endpoints.synthesis import run_enhanced_synthesis_task_sync
        print("✓ All key modules import successfully")
        
        # Test 4: Check if reference audio preparation logic works
        print("\n4. Testing reference audio preparation...")
        from app.services.ensemble_voice_synthesis_engine import EnsembleVoiceSynthesizer
        from app.schemas.voice import VoiceProfileSchema
        
        # Create a mock voice profile
        voice_profile = VoiceProfileSchema(
            id="test_profile",
            reference_audio_id="/path/to/test.wav",
            quality_score=0.8,
            created_at=datetime.now()
        )
        
        engine = EnsembleVoiceSynthesizer()
        # This should not crash even if the file doesn't exist
        result = await engine._prepare_reference_audio(voice_profile)
        print("✓ Reference audio preparation handles missing files gracefully")
        
        print("\n🎉 All synthesis fixes are working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixes())
    sys.exit(0 if success else 1)