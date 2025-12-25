# Voice Replication System

A simplified, working voice replication system with real TTS capabilities.

## Features

- **File Upload**: Upload audio files (MP3, MP4, WAV) for voice analysis
- **Voice Analysis**: Extract voice characteristics from reference audio
- **Text Processing**: Validate and process text input with language detection
- **Speech Synthesis**: Generate speech using voice cloning (simplified version)
- **Session Management**: Secure session handling for user data isolation
- **Real-time Progress**: Track processing progress for all operations

## Quick Start

### Prerequisites

- Python 3.11+ with conda
- Node.js 18+ with yarn
- FFmpeg (for audio processing)

### Installation

1. **Create and activate conda environment:**
   ```bash
   conda create -n ey2 python=3.11 -y
   conda activate ey2
   ```

2. **Install backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies:**
   ```bash
   cd frontend
   yarn install
   ```

### Running the Application

1. **Start the backend server:**
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
   ```

2. **Start the frontend development server:**
   ```bash
   cd frontend
   yarn dev
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8001
   - API Documentation: http://localhost:8001/docs

### Testing the APIs

You can test the APIs using the following endpoints:

- **Health Check**: `GET http://localhost:8001/health`
- **Create Session**: `POST http://localhost:8001/api/v1/session/create`
- **Validate Text**: `POST http://localhost:8001/api/v1/text/validate`
- **Upload File**: `POST http://localhost:8001/api/v1/files/upload`
- **Analyze Voice**: `POST http://localhost:8001/api/v1/voice/analyze`
- **Synthesize Speech**: `POST http://localhost:8001/api/v1/synthesis/synthesize`

### Test Files

Use the audio files in the `downloads/` folder for testing:
- `Taylor Swift - The Fate of Ophelia (Official Music Video).mp3`
- `Taylor Swift - The Fate of Ophelia (Official Music Video).mp4`

## Architecture

### Backend (FastAPI)
- **Core Services**: Real voice synthesis, file processing, session management
- **Database**: SQLite with SQLAlchemy ORM
- **Audio Processing**: librosa, soundfile for audio analysis
- **Voice Synthesis**: Simplified synthesis service (can be upgraded to full TTS)

### Frontend (Next.js)
- **React Components**: Modern UI with TypeScript
- **API Client**: Axios-based client with error handling
- **Real-time Updates**: Progress tracking and status updates
- **Responsive Design**: Works on desktop and mobile

## Development Notes

- The current implementation uses a simplified voice synthesis service for development
- All mock implementations have been removed
- The system is designed to be easily upgraded with full TTS capabilities
- Session management ensures data isolation between users
- Comprehensive error handling and logging throughout

## Production Deployment

For production deployment:
1. Set up proper environment variables
2. Configure a production database (PostgreSQL recommended)
3. Set up Redis for caching and task queues
4. Configure proper CORS settings
5. Set up SSL/TLS certificates
6. Use a production WSGI server (gunicorn)
7. Set up monitoring and logging

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure all APIs work correctly
5. Test both frontend and backend integration

## License

This project is for educational and development purposes.