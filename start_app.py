#!/usr/bin/env python3
"""
Startup script for Voice Style Replication application.
Starts both backend and frontend with correct port configurations.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_backend():
    """Start the backend server on port 8000."""
    print("🚀 Starting backend server on port 8000...")
    
    backend_dir = Path(__file__).parent / "backend"
    
    # Change to backend directory and start uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app", 
        "--host", "0.0.0.0",
        "--port", "8000", 
        "--reload"
    ]
    
    return subprocess.Popen(
        cmd, 
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

def start_frontend():
    """Start the frontend server on port 3000."""
    print("🌐 Starting frontend server on port 3000...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Start Next.js development server
    cmd = ["npm", "run", "dev"]
    
    return subprocess.Popen(
        cmd,
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

def main():
    """Main startup function."""
    print("🎤 Voice Style Replication - Application Startup")
    print("=" * 50)
    
    try:
        # Start backend
        backend_process = start_backend()
        
        # Wait a bit for backend to start
        print("⏳ Waiting for backend to initialize...")
        time.sleep(5)
        
        # Start frontend
        frontend_process = start_frontend()
        
        print("\n✅ Application started successfully!")
        print("📍 Backend:  http://localhost:8000")
        print("📍 Frontend: http://localhost:3000")
        print("\n💡 Press Ctrl+C to stop both servers")
        
        # Monitor processes
        try:
            while True:
                # Check if processes are still running
                backend_running = backend_process.poll() is None
                frontend_running = frontend_process.poll() is None
                
                if not backend_running:
                    print("❌ Backend process stopped unexpectedly")
                    break
                    
                if not frontend_running:
                    print("❌ Frontend process stopped unexpectedly")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down servers...")
            
            # Terminate processes
            backend_process.terminate()
            frontend_process.terminate()
            
            # Wait for clean shutdown
            backend_process.wait(timeout=10)
            frontend_process.wait(timeout=10)
            
            print("✅ Servers stopped successfully")
            
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()