"""
Simple test for voice model training system.
"""

import asyncio
from pathlib import Path
from app.services.intelligent_voice_model_trainer import IntelligentVoiceModelTrainer


async def test_simple_model_creation():
    """Test simple model creation with real audio."""
    print("Testing simple voice model creation...")
    
    # Use real audio file from downloads folder
    downloads_dir = Path("../downloads")
    audio_file = downloads_dir / "Taylor Swift - The Fate of Ophelia (Official Music Video).mp3"
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return False
    
    print(f"Using audio file: {audio_file}")
    
    trainer = IntelligentVoiceModelTrainer()
    
    def progress_callback(progress, message):
        print(f"  Progress: {progress}% - {message}")
    
    try:
        # Test with just one audio file (duplicated to meet segment requirements)
        audio_paths = [str(audio_file)]
        
        success, model_id, metadata = await trainer.create_dedicated_voice_model(
            audio_paths=audio_paths,
            voice_profile_id="test_profile_simple",
            progress_callback=progress_callback
        )
        
        if success:
            print(f"✅ Successfully created model: {model_id}")
            print(f"   Quality score: {metadata.get('quality_score', 0):.2f}")
            print(f"   Training duration: {metadata.get('training_duration', 0):.1f}s")
            return True
        else:
            print(f"❌ Model creation failed: {metadata}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run simple test."""
    print("Simple Voice Model Training Test")
    print("=" * 40)
    
    success = await test_simple_model_creation()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)