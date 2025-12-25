#!/usr/bin/env python3
"""
Test script to verify that the timeout fix works for voice synthesis.
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path

API_BASE_URL = "http://localhost:8001"

async def test_synthesis_timeout():
    """Test that synthesis requests don't timeout."""
    print("Testing synthesis timeout fix...")
    
    # First, check if the API is available
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_BASE_URL}/api/v1/status") as response:
                if response.status == 200:
                    print("✅ API is accessible")
                else:
                    print(f"❌ API returned status {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False
    
    # Test synthesis request with extended timeout
    synthesis_request = {
        "text": "This is a test of the voice synthesis system with extended timeout configuration.",
        "voice_model_id": "test_voice_model",
        "language": "en",
        "voice_settings": {
            "pitch_shift": 0,
            "speed_factor": 1.0,
            "emotion_intensity": 0.5,
            "volume_gain": 0
        },
        "output_format": "wav",
        "quality": "high"
    }
    
    print("Starting synthesis request...")
    start_time = time.time()
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        try:
            async with session.post(
                f"{API_BASE_URL}/api/v1/synthesis/synthesize",
                json=synthesis_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                elapsed = time.time() - start_time
                print(f"Response received after {elapsed:.2f} seconds")
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Synthesis request successful: {result}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Synthesis request failed: {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Request timed out after {elapsed:.2f} seconds")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Request failed after {elapsed:.2f} seconds: {e}")
            return False

async def main():
    """Main test function."""
    print("Voice Synthesis Timeout Fix Test")
    print("=" * 40)
    
    success = await test_synthesis_timeout()
    
    if success:
        print("\n✅ Timeout fix test PASSED")
        print("The synthesis endpoint should now handle longer processing times.")
    else:
        print("\n❌ Timeout fix test FAILED")
        print("The timeout configuration may need further adjustment.")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)