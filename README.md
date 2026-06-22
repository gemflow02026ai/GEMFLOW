# Jewel Mission — FastAPI Backend

AI-powered image processing backend for the Jewel Mission Flutter app.

## Quick Start

### 1. Setup Python environment
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
copy .env.example .env
# Edit .env with your actual API keys
```

### 4. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/process-ecommerce` | Process 3 e-commerce images (OpenAI) |
| POST | `/process-creative` | Process 1 creative image (Grok) |

## API Docs

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Flutter Connection

For **Android Emulator**, the app defaults to `http://10.0.2.2:8000` which maps to `localhost` on the host machine.

For **Physical Device**, update the `baseUrl` in `lib/main.dart`:
```dart
ApiService(baseUrl: 'http://YOUR_COMPUTER_IP:8000')
```

## Prototype Mode

If API keys are not configured, the backend runs in **pass-through mode** — it accepts images and returns them unmodified. This allows testing the full app flow without AI services.
