# Deployment

The app is a small Flask service. All disease diagnosis runs on an **Ollama
vision model over HTTP** (Ollama Cloud by default), so there is no bundled ML
model and the container image stays small.

## 1. Configure

Copy the example env file and fill in your Ollama Cloud key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=your_ollama_cloud_key
OLLAMA_MODEL=gemma4:31b
```

`.env` is git-ignored and Docker-ignored, so your key is never committed or
baked into the image — it is injected at runtime.

## 2. Run with Docker Compose (recommended)

```bash
docker compose up --build -d
```

The app is then available at http://localhost:5000

Stop it with:

```bash
docker compose down
```

## 3. Or run with plain Docker

```bash
docker build -t crop-disease-detection .
docker run -d -p 5000:5000 --env-file .env --name crop-disease-detection crop-disease-detection
```

## 4. Deploy to a host / PaaS

- The image listens on `$PORT` (default `5000`), so it works on platforms that
  inject a port (Render, Railway, Fly.io, Cloud Run, etc.).
- Set `OLLAMA_HOST`, `OLLAMA_API_KEY`, and `OLLAMA_MODEL` as environment
  variables in the platform's dashboard — do **not** commit them.
- A `render.yaml` is included for one-click Render deploys; set
  `OLLAMA_API_KEY` (and `OLLAMA_HOST` if not cloud) in the Render dashboard.

## Notes

- Uploaded images are stored in `static/shots`. In Compose they persist in the
  `uploads` volume; plain `docker run` keeps them only for the container's life.
- The container runs `gunicorn` as a non-root user with a `/` health check.
