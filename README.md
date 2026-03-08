# Place Names Voice Recorder

A simple web app for collecting labeled voice recordings of location names. Recordings are stored as WAV (16 kHz, mono, 16-bit) in `dataset/<Location_Name>/` and tracked in `dataset/metadata.csv` for later ML use.

## Setup

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

## Usage

1. Enter a **location name** (e.g. "Mosque Central", "Hospital A"). Spaces become underscores in folder names.
2. Click **Start Recording**, speak the place name (about 2–5 seconds), then **Stop Recording**.
3. Click **Save Recording** to upload. The file is converted to WAV and saved as `dataset/<Location>_<NNN>.wav` and appended to `dataset/metadata.csv`.

## Project layout

```
project/
  app.py
  requirements.txt
  README.md
  templates/
    index.html
  static/
    script.js
    style.css
  dataset/
    metadata.csv
    Mosque_Central/
      Mosque_Central_001.wav
      ...
```

## Requirements

- **Microphone**: browser will ask for permission; recording is blocked if denied.
- **Location required**: you cannot save without entering a location name.
- **ffmpeg**: must be on your PATH for the server to convert uploaded audio to WAV.
