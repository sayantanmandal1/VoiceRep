#!/usr/bin/env python3
"""
Script to clear corrupted Bark model cache.
"""

import shutil
import os
from pathlib import Path

def clear_bark_cache():
    """Clear corrupted Bark model cache."""
    try:
        # Check both possible cache locations
        cache_locations = [
            Path.home() / ".cache" / "tts",
            Path.home() / "AppData" / "Local" / "tts"
        ]
        
        removed_count = 0
        
        for cache_dir in cache_locations:
            print(f"Checking cache directory: {cache_dir}")
            
            if not cache_dir.exists():
                print(f"Cache directory does not exist: {cache_dir}")
                continue
            
            # Patterns to look for Bark model cache
            bark_patterns = [
                "*bark*",
                "*multilingual*multi-dataset*bark*",
                "tts_models--multilingual--multi-dataset--bark"
            ]
            
            for pattern in bark_patterns:
                cache_dirs = list(cache_dir.glob(pattern))
                for cache_dir_path in cache_dirs:
                    if cache_dir_path.is_dir():
                        print(f"Removing corrupted Bark cache: {cache_dir_path}")
                        try:
                            shutil.rmtree(cache_dir_path, ignore_errors=True)
                            removed_count += 1
                            print(f"✓ Removed: {cache_dir_path}")
                        except Exception as e:
                            print(f"✗ Failed to remove {cache_dir_path}: {e}")
            
            # Also check for specific Bark model directory
            bark_specific_dir = cache_dir / "tts_models--multilingual--multi-dataset--bark"
            if bark_specific_dir.exists():
                print(f"Removing specific Bark model directory: {bark_specific_dir}")
                try:
                    shutil.rmtree(bark_specific_dir, ignore_errors=True)
                    removed_count += 1
                    print(f"✓ Removed: {bark_specific_dir}")
                except Exception as e:
                    print(f"✗ Failed to remove {bark_specific_dir}: {e}")
        
        if removed_count > 0:
            print(f"\n✅ Successfully removed {removed_count} corrupted cache directories.")
            print("The Bark model will be re-downloaded on next startup.")
        else:
            print("\n📁 No Bark cache directories found to remove.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error clearing Bark cache: {e}")
        return False

if __name__ == "__main__":
    print("🧹 Clearing corrupted Bark model cache...")
    success = clear_bark_cache()
    
    if success:
        print("\n🎉 Cache clearing completed successfully!")
        print("You can now restart the server to re-download the Bark model.")
    else:
        print("\n💥 Cache clearing failed. Please check the error messages above.")