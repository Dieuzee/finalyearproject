# System Architecture

## 1. Architectural style

BID CROP follows a classic **three-tier client–server web architecture**, with
the AI diagnosis delegated to an **external cloud service**. Inside the web
server it applies a lightweight **MVC-style** separation of concerns:

- **Controllers** — the Flask routes in `app.py`
- **Views** — the Jinja2 HTML templates
- **Model / data** — `db.py` and the normalised result produced from the AI response

Heavy machine-learning work is performed off-device by a hosted vision model, so
the application itself remains small, stateless and easy to deploy.

## 2. The three tiers

### 2.1 Presentation tier (client / browser)

The user's web browser renders the Jinja2 templates:

- `index.html` — landing page
- `input.html` — image upload and live camera capture
- `display.html` — diagnosis results (also used to reopen a saved scan)
- `history.html` — saved scan history, statistics, search/filter
- `base.html` — shared header/navigation

Client-side JavaScript handles drag-and-drop upload, live camera capture via the
browser `getUserMedia` API, the responsive mobile menu, and small UI
enhancements. **Responsibility:** capture the leaf image and present the results.

### 2.2 Application tier (Flask web server — `app.py`)

The core of the system. It is organised into three concerns:

1. **Routing / controllers** — `/`, `/input`, `/upload`, `/display`,
   `/history`, `/scan/<id>`, and the delete/clear routes.
2. **Business logic** — file-type validation, saving uploads, building the
   structured prompt (`ANALYSIS_PROMPT`), calling the AI, parsing and
   normalising the JSON response (`_parse_json`, `analyze_with_ollama`), the
   leaf / non-leaf gate, and an in-memory cache of the most recent result.
3. **Integration layer** — the `ollama` client, configured to talk to Ollama
   Cloud over HTTPS.

### 2.3 Data tier

- **SQLite database** (`db.py`, `scans.db`) — persists every diagnosis in the
  `scans` table.
- **File storage** (`static/shots/`) — the uploaded image files, referenced by
  filename from the database.

## 3. External service

The actual AI is an **Ollama Cloud vision model** (`qwen2.5vl:7b-cloud`). The
Flask server sends it the image plus a structured prompt over HTTPS and receives
a JSON diagnosis in return. It is a third-party dependency selected purely by
configuration, not part of the application code.

## 4. Request flow

### 4.1 Diagnosis

```
Browser                Flask (app.py)                      Data / External
  |  upload image  -->  /upload
  |                     |- validate + save file  ------>   static/shots/
  |                     |- analyze_with_ollama()  ------>   Ollama Cloud (HTTPS)
  |                     |         <-- JSON diagnosis ---
  |                     |- normalise + cache result
  |                     |- db.save_scan()         ------>   scans.db (SQLite)
  |  <-- results page - display.html
```

### 4.2 History

```
Browser -- GET /history ---> Flask -- db.query_scans() / get_stats() --> SQLite --> history.html
Browser -- GET /scan/<id> -> Flask -- db.get_scan() ------------------> SQLite --> display.html
```

## 5. Deployment view

- Packaged as a **Docker** container (`Dockerfile`).
- Served in production by **gunicorn** (a WSGI server).
- Deployable to **Render** via `render.yaml`.
- Configuration (`OLLAMA_HOST`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`) is injected at
  runtime through environment variables / a `.env` file, so no secrets are baked
  into the image.

## 6. Key architectural characteristics

- **Stateless request handling** — apart from a small last-result cache, each
  request is self-contained, which keeps the server simple and horizontally
  scalable.
- **Separation of concerns** — the ML workload runs in the cloud, so the app has
  no bundled model and stays lightweight.
- **Loose coupling to the AI** — the model and host are chosen by configuration;
  switching models or providers requires no code change.
- **Layered structure** — presentation, application and data responsibilities
  are clearly separated, which aids testing and maintenance.
