# Voice Style Replication Application

A full-stack application for high-fidelity voice cloning and synthesis. Upload audio or video files to clone voices and generate new speech content while preserving the original speaker's vocal characteristics.

## Features

- **Voice Cloning**: Extract and replicate voice characteristics from audio/video files
- **Multi-format Support**: Process .mp3, .wav, .flac, .m4a audio and .mp4, .avi, .mov, .mkv video files
- **Cross-language Synthesis**: Generate speech in different languages while preserving voice characteristics
- **Real-time Processing**: Background task processing with progress tracking
- **High-quality Output**: Generate natural-sounding speech with preserved prosody and emotional tone

## Architecture

- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Backend**: FastAPI with Python 3.11+
- **Task Queue**: Celery with Redis
- **Database**: SQLite with Alembic migrations
- **AI Models**: TorToiSe TTS and Real-Time Voice Cloning (RVC)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for local development)

### Docker Setup (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd voice-style-replication
```

2. Start all services:
```bash
docker-compose up --build
```

3. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Task Monitor (Flower): http://localhost:5555

### Local Development Setup

#### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment:
```bash
cp .env.example .env
```

5. Initialize database:
```bash
alembic upgrade head
```

6. Start Redis (required for task queue):
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

7. Start backend server:
```bash
uvicorn app.main:app --reload
```

8. Start Celery worker (in another terminal):
```bash
celery -A app.core.celery_app worker --loglevel=info
```

#### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

## Project Structure

```
voice-style-replication/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core configuration
│   │   ├── models/         # Database models
│   │   ├── tasks/          # Celery background tasks
│   │   └── main.py         # FastAPI application
│   ├── alembic/            # Database migrations
│   ├── tests/              # Backend tests
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── app/               # Next.js app directory
│   ├── components/        # React components (to be created)
│   └── package.json       # Node.js dependencies
├── uploads/               # File upload storage
├── models/                # AI model storage
├── results/               # Generated audio storage
└── docker-compose.yml     # Docker services configuration
```

## Development Workflow

This project follows a spec-driven development approach. The implementation is organized into tasks that can be executed incrementally:

1. **View Tasks**: Open `.kiro/specs/voice-style-replication/tasks.md`
2. **Execute Tasks**: Click "Start task" next to each task item
3. **Track Progress**: Monitor task completion and test results

## API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation.

## Testing

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Contributing

1. Follow the task-based development workflow
2. Ensure all tests pass before submitting changes
3. Update documentation as needed

## License

[Add your license information here]