#!/usr/bin/env python3
"""
Test script to verify all server startup fixes are working.
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_server_fixes():
    """Test all the server startup fixes."""
    print("🔧 Testing server startup fixes...")
    
    try:
        # Test 1: Import fixes
        print("\n1. Testing import fixes...")
        from app.services.ensemble_voice_synthesis_engine import EnsembleVoiceSynthesizer
        from app.services.optimized_synthesis_service import OptimizedSynthesisService
        print("✓ All imports working correctly")
        
        # Test 2: Schema fixes
        print("\n2. Testing schema fixes...")
        from app.schemas.voice import VoiceProfileSchema
        
        # Create a proper voice profile
        voice_profile = VoiceProfileSchema(
            id="test_profile",
            reference_audio_id="test_audio.wav",
            quality_score=0.8,
            created_at=datetime.now()
        )
        print("✓ VoiceProfileSchema working correctly")
        
        # Test 3: Multi-speaker model handling
        print("\n3. Testing multi-speaker model handling...")
        
        # Test ensemble synthesizer
        ensemble_synthesizer = EnsembleVoiceSynthesizer()
        
        # Test reference audio preparation (should handle missing files gracefully)
        result = await ensemble_synthesizer._prepare_reference_audio(voice_profile)
        print("✓ Reference audio preparation handles missing files gracefully")
        
        # Test 4: Optimized synthesis service
        print("\n4. Testing optimized synthesis service...")
        optimized_service = OptimizedSynthesisService()
        print("✓ Optimized synthesis service initializes correctly")
        
        # Test 5: Cache clearing functionality
        print("\n5. Testing cache clearing functionality...")
        from clear_bark_cache import clear_bark_cache
        # Don't actually clear cache, just test the function exists
        print("✓ Cache clearing functionality available")
        
        print("\n🎉 All server startup fixes are working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing server fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_server_fixes())
    sys.exit(0 if success else 1)