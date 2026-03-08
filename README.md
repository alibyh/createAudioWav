# Place Names Voice Recorder

A simple web app for collecting labeled voice recordings of location names. Recordings are stored as WAV (16 kHz, mono, 16-bit) in `dataset/<Location_Name>/` and tracked in `dataset/metadata.csv` for later ML use.

You can run it **locally** or deploy **frontend on GitHub Pages** and **backend on Render** so everyone who visits the page saves into one shared dataset (optionally stored in this repo).

---

## Local setup

1. **Python 3.8+** and **ffmpeg** (required for converting browser audio to WAV).

   ```bash
   # macOS (Homebrew)
   brew install ffmpeg
   ```

2. Create a virtualenv and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

4. Open **http://127.0.0.1:5000** in a modern browser (Chrome/Firefox/Edge). Allow microphone access when prompted.

---

## Deploy for everyone (GitHub Pages + shared dataset)

Goal: **Frontend** on GitHub Pages; **backend** on Render. Everyone who hits “Save” sends recordings to the same backend, and you can store the dataset in this GitHub repo.

### 1. Deploy the backend (Render)

1. Push this repo to GitHub (if you haven’t already).
2. Go to [render.com](https://render.com), sign in with GitHub, and create a **New → Web Service**.
3. Connect the repo and set:
   - **Runtime**: Docker (the repo includes a `Dockerfile` with Python + ffmpeg).
   - **Build**: leave default (uses `Dockerfile`).
   - **Instance type**: Free (or paid if you prefer).
4. Under **Environment** add:
   - **CORS_ORIGIN**  
     - For “any site can use this API”: leave empty or set `*` (default).  
     - To allow only your GitHub Pages site: `https://YOUR_USERNAME.github.io` or `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`.
   - **GITHUB_REPO** (optional but recommended): `YOUR_USERNAME/YOUR_REPO_NAME`.  
     If set, each saved recording is **pushed to this repo** under `dataset/<Location>/<file>.wav` and `dataset/metadata.csv`, so the dataset lives in the repo and survives backend restarts.
   - **GITHUB_TOKEN** (required if you set **GITHUB_REPO**): a [Personal Access Token](https://github.com/settings/tokens) with scope **repo** (so the backend can create/update files in the repo).
5. Deploy. Note the service URL, e.g. `https://voice-recorder-api-xxxx.onrender.com`.

### 2. Point the frontend to the backend

1. In **docs/index.html**, set your backend URL:
   ```html
   <script>
       window.API_BASE = "https://voice-recorder-api-xxxx.onrender.com";
   </script>
   ```
   Replace with your real Render URL (no trailing slash).

2. Commit and push.

### 3. Turn on GitHub Pages

1. In the repo: **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Branch: **main** (or **master**), folder: **/docs**.
4. Save. The site will be at `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`.

Anyone visiting that URL can record and click **Save All**; recordings go to your Render backend and, if **GITHUB_REPO** and **GITHUB_TOKEN** are set, are stored in this repo under `dataset/`.

### Where are my files?

- **If you did not set GITHUB_REPO and GITHUB_TOKEN on Render:** The backend only writes to its own disk. On Render’s free tier that disk is **ephemeral** (wiped on sleep/restart), so files disappear. You’ll see “Saved” but they won’t appear on GitHub or anywhere persistent.
- **To have files in your GitHub repo:** In the Render dashboard for your service, go to **Environment** and add:
  - **GITHUB_REPO** = `YOUR_USERNAME/YOUR_REPO_NAME` (the repo that contains this project).
  - **GITHUB_TOKEN** = a [Personal Access Token](https://github.com/settings/tokens) with **repo** scope.
- After saving again, files appear in the repo under **dataset/** → **dataset/\<LocationName\>/\<file\>.wav** and **dataset/metadata.csv**. If something goes wrong, the app now shows the reason (e.g. “GITHUB_REPO not set” or the GitHub API error) under the success message.

---

## Usage

1. Enter a **location name** (e.g. "Mosque Central", "Hospital A"). Spaces become underscores in folder names.
2. Click **Start Recording**, speak the place name (about 2–5 seconds), then **Stop Recording**. You can record up to 5 clips.
3. Click **Save All** to upload. Files are converted to WAV and saved as `dataset/<Location>_<NNN>.wav` and appended to `dataset/metadata.csv`.

---

## Project layout

```
  app.py              # Flask backend
  Dockerfile          # For Render (Python + ffmpeg)
  render.yaml         # Optional Render blueprint
  requirements.txt
  README.md
  templates/
    index.html
  static/
    script.js
    style.css
  docs/               # Static site for GitHub Pages
    index.html        # Set window.API_BASE here
    static/
      style.css
      script.js
  dataset/
    metadata.csv
    Mosque_Central/
      Mosque_Central_001.wav
      ...
```

---

## Requirements

- **Microphone**: browser will ask for permission; recording is blocked if denied.
- **Location required**: you cannot save without entering a location name.
- **ffmpeg**: must be on your PATH for the server to convert uploaded audio to WAV (included in the Docker image for Render).
