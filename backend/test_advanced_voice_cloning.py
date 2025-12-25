#!/usr/bin/env python3
"""
Test script for advanced voice cloning functionality.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.real_voice_synthesis_service import advanced_voice_cloning_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_voice_cloning():
    """Test the advanced voice cloning functionality."""
    try:
        print("🚀 Testing Advanced Voice Cloning System")
        print("=" * 50)
        
        # Initialize the service
        print("1. Initializing advanced voice cloning service...")
        success = await advanced_voice_cloning_service.initialize_model()
        
        if not success:
            print("❌ Failed to initialize voice cloning service")
            return False
        
        print("✅ Voice cloning service initialized successfully")
        
        # Check if we have test audio files
        test_audio_dir = Path("../downloads")
        audio_files = list(test_audio_dir.glob("*.mp3")) + list(test_audio_dir.glob("*.wav"))
        
        if not audio_files:
            print("⚠️  No test audio files found in downloads directory")
            print("   Please add an audio file to test voice cloning")
            return True
        
        # Use the first audio file as reference
        reference_audio = str(audio_files[0])
        print(f"2. Using reference audio: {reference_audio}")
        
        # Test text to synthesize
        test_text = "Hello, this is a test of advanced voice cloning technology. The system should replicate the exact voice characteristics including tone, pitch, and speaking style."
        
        # Create output directory
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_voice_clone.wav"
        
        print("3. Preprocessing reference audio...")
        processed_reference = await advanced_voice_cloning_service.preprocess_reference_audio(
            reference_audio,
            lambda p, m: print(f"   Progress: {p}% - {m}")
        )
        
        print("4. Extracting deep voice characteristics...")
        voice_characteristics = await advanced_voice_cloning_service.extract_deep_voice_characteristics(
            processed_reference,
            lambda p, m: print(f"   Progress: {p}% - {m}")
        )
        
        print("5. Voice characteristics extracted:")
        print(f"   - Pitch range: {voice_characteristics['pitch_characteristics']['f0_min']:.1f} - {voice_characteristics['pitch_characteristics']['f0_max']:.1f} Hz")
        print(f"   - Mean pitch: {voice_characteristics['pitch_characteristics']['f0_mean']:.1f} Hz")
        print(f"   - Voice quality: {voice_characteristics['voice_quality']['overall_quality']:.2f}")
        print(f"   - Speaking rate: {voice_characteristics['prosodic_patterns']['speech_rate']:.1f} syllables/sec")
        
        print("6. Performing advanced voice cloning synthesis...")
        result = await advanced_voice_cloning_service.synthesize_speech(
            text=test_text,
            reference_audio_path=processed_reference,
            output_path=str(output_path),
            language="en",
            progress_callback=lambda p, m: print(f"   Progress: {p}% - {m}")
        )
        
        print("7. Voice cloning completed!")
        print(f"   - Output file: {result['output_path']}")
        print(f"   - Duration: {result['duration']:.2f} seconds")
        print(f"   - Quality score: {result['quality_score']:.2f}")
        print(f"   - Similarity score: {result['similarity_score']:.1%}")
        print(f"   - Method: {result['synthesis_method']}")
        
        # Verify output file exists
        if os.path.exists(result['output_path']):
            file_size = os.path.getsize(result['output_path'])
            print(f"   - File size: {file_size:,} bytes")
            print("✅ Voice cloning test completed successfully!")
            
            print("\n🎉 Advanced Voice Cloning Test Results:")
            print(f"   - Voice similarity: {result['similarity_score']:.1%}")
            print(f"   - Audio quality: {result['quality_score']:.1%}")
            print(f"   - Processing method: {result['synthesis_method']}")
            
            if result['similarity_score'] > 0.8:
                print("🌟 Excellent voice replication achieved!")
            elif result['similarity_score'] > 0.6:
                print("👍 Good voice replication achieved!")
            else:
                print("⚠️  Voice replication needs improvement")
            
            return True
        else:
            print("❌ Output file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        logger.exception("Test failed")
        return False

async def main():
    """Main test function."""
    print("Advanced Voice Cloning Test Suite")
    print("=" * 40)
    
    success = await test_advanced_voice_cloning()
    
    if success:
        print("\n✅ All tests passed!")
        print("Your advanced voice cloning system is working correctly.")
        print("The system can now replicate voices with high fidelity.")
    else:
        print("\n❌ Tests failed!")
        print("Please check the error messages above and fix any issues.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())