"""
Flask backend for collecting labeled voice recordings of place names.
Stores WAV files in dataset/<location_folder>/ and tracks metadata in dataset/metadata.csv.
Supports CORS for GitHub Pages; optional push to GitHub repo for persistent shared dataset.
"""

import base64
import csv
import os
from pathlib import Path
from typing import Tuple

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
METADATA_PATH = DATASET_DIR / "metadata.csv"
PLACES_PATH = DATASET_DIR / "places_names.json"
MAX_UPLOAD_MB = 10

# CORS: allow frontend on GitHub Pages (or any origin) to call this API
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "").strip()  # e.g. "username/repo-name"


def location_to_folder_name(location: str) -> str:
    """Convert location display name to folder name (spaces -> underscores)."""
    return location.strip().replace(" ", "_") if location else ""


def ensure_dataset_structure():
    """Create dataset dir and metadata.csv if they don't exist."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if not METADATA_PATH.exists():
        with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "location"])


def get_next_index(folder_path: Path, prefix: str) -> int:
    """Return next available index for files like <prefix>_001.wav."""
    if not folder_path.exists():
        return 1
    existing = list(folder_path.glob(f"{prefix}_*.wav"))
    if not existing:
        return 1
    indices = []
    for f in existing:
        try:
            # Mosque_Central_001.wav -> 1
            num = int(f.stem.split("_")[-1])
            indices.append(num)
        except (ValueError, IndexError):
            continue
    return max(indices, default=0) + 1


def ensure_wav_16k_mono_16bit(input_path: Path, output_path: Path) -> None:
    """Convert audio file to WAV 16kHz mono 16-bit. Uses pydub if available."""
    try:
        from pydub import AudioSegment
    except ImportError:
        # No pydub: if input is already WAV, copy; else raise
        import shutil
        if input_path.suffix.lower() == ".wav":
            shutil.copy2(input_path, output_path)
            return
        raise RuntimeError(
            "pydub is not installed in this Python environment. "
            "Activate your virtual environment (source venv/bin/activate), then run: pip install pydub"
        )

    sound = AudioSegment.from_file(str(input_path))
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)  # 16-bit
    sound.export(str(output_path), format="wav", parameters=["-ac", "1", "-ar", "16000"])


def append_metadata(filename: str, location: str) -> None:
    """Append one row to dataset/metadata.csv."""
    ensure_dataset_structure()
    with open(METADATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([filename, location])


def push_file_to_github(path_in_repo: str, content_bytes: bytes, message: str) -> Tuple[bool, str]:
    """Create or update a file in the GitHub repo. Returns (success, error_message)."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False, "GITHUB_REPO or GITHUB_TOKEN not set on server"
    try:
        import urllib.request
        import json
        from urllib.error import HTTPError
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path_in_repo}"
        b64 = base64.standard_b64encode(content_bytes).decode("ascii")
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        try:
            with urllib.request.urlopen(req) as r:
                existing = json.loads(r.read().decode())
                sha = existing.get("sha")
        except HTTPError as e:
            if e.code != 404:
                return False, f"GitHub API {e.code}: {e.reason}"
            sha = None
        payload = {"message": message, "content": b64}
        if sha is not None:
            payload["sha"] = sha
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req):
            return True, ""
    except HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return False, f"GitHub API {e.code}: {body[:200] if body else e.reason}"
    except Exception as e:
        return False, str(e)


def get_next_index_github(folder_name: str) -> int:
    """Get next index by listing dataset/<folder_name>/ in the GitHub repo."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return 1
    try:
        import urllib.request
        import json
        from urllib.error import HTTPError
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/dataset/{folder_name}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        with urllib.request.urlopen(req) as r:
            items = json.loads(r.read().decode())
        indices = []
        prefix = f"{folder_name}_"
        for item in items:
            if item.get("type") != "file":
                continue
            name = item.get("name", "")
            if name.startswith(prefix) and name.endswith(".wav"):
                try:
                    num = int(name[len(prefix) : -4])
                    indices.append(num)
                except ValueError:
                    pass
        return max(indices, default=0) + 1
    except HTTPError as e:
        if e.code == 404:
            return 1
        return 1
    except Exception:
        return 1


def push_metadata_line_to_github(filename: str, location: str) -> Tuple[bool, str]:
    """Append a line to metadata.csv in the repo. Returns (success, error_message)."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False, "GITHUB_REPO or GITHUB_TOKEN not set"
    path_in_repo = "dataset/metadata.csv"
    try:
        import urllib.request
        import json
        from urllib.error import HTTPError
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path_in_repo}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        try:
            with urllib.request.urlopen(req) as r:
                existing = json.loads(r.read().decode())
                current = base64.standard_b64decode(existing["content"]).decode("utf-8")
                if not current.endswith("\n"):
                    current += "\n"
                new_content = current + f"{filename},{location}\n"
        except HTTPError as e:
            if e.code == 404:
                new_content = "filename,location\n" + f"{filename},{location}\n"
            else:
                return False, f"GitHub API {e.code}: {e.reason}"
        return push_file_to_github(
            path_in_repo,
            new_content.encode("utf-8"),
            f"Add recording: {filename}",
        )
    except Exception as e:
        return False, str(e)


@app.after_request
def add_cors_headers(response):
    """Allow frontend on GitHub Pages (or other origins) to call the API."""
    if request.path.startswith("/api/"):
        response.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/upload", methods=["OPTIONS"])
@app.route("/api/recordings/count", methods=["OPTIONS"])
def options_cors():
    return "", 204


def load_places_names() -> list:
    """Load list of place names from dataset/places_names.json (local or from GitHub)."""
    import json as json_mod
    # Try local file first
    if PLACES_PATH.exists():
        try:
            with open(PLACES_PATH, "r", encoding="utf-8") as f:
                data = json_mod.load(f)
            if isinstance(data, list):
                return [item.get("name", "").strip() for item in data if isinstance(item, dict) and item.get("name")]
            return []
        except Exception:
            return []
    # Fallback: fetch from GitHub if configured
    if GITHUB_TOKEN and GITHUB_REPO:
        try:
            import urllib.request
            url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/dataset/places_names.json"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
            req.add_header("Accept", "application/vnd.github+json")
            req.add_header("X-GitHub-Api-Version", "2022-11-28")
            with urllib.request.urlopen(req) as r:
                obj = json_mod.loads(r.read().decode())
            content = base64.standard_b64decode(obj.get("content", "")).decode("utf-8")
            data = json_mod.loads(content)
            if isinstance(data, list):
                return [item.get("name", "").strip() for item in data if isinstance(item, dict) and item.get("name")]
        except Exception:
            pass
    return []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/places", methods=["GET"])
def get_places():
    """Return list of place names for the dropdown (from dataset/places_names.json)."""
    places = load_places_names()
    return jsonify({"places": places})


def get_recording_count_github(folder_name: str) -> int:
    """Return number of WAV files in dataset/<folder_name>/ from GitHub repo."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return 0
    try:
        import urllib.request
        import json
        from urllib.error import HTTPError
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/dataset/{folder_name}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        with urllib.request.urlopen(req) as r:
            items = json.loads(r.read().decode())
        return sum(1 for i in items if i.get("type") == "file" and (i.get("name") or "").endswith(".wav"))
    except HTTPError as e:
        if e.code == 404:
            return 0
        return 0
    except Exception:
        return 0


@app.route("/api/recordings/count", methods=["GET"])
def get_recording_count():
    """Return count of recordings for a location (folder name with underscores)."""
    folder_name = request.args.get("location_folder", "").strip()
    if not folder_name:
        return jsonify({"count": 0})
    if GITHUB_REPO:
        count = get_recording_count_github(folder_name)
    else:
        folder_path = DATASET_DIR / folder_name
        count = len(list(folder_path.glob("*.wav"))) if folder_path.is_dir() else 0
    return jsonify({"count": count})


@app.route("/api/upload", methods=["POST"])
def upload_recording():
    """
    Receive one or more audio blobs and location.
    Create folder, convert each to WAV with auto-increment names, update metadata.
    """
    ensure_dataset_structure()

    location = (request.form.get("location") or "").strip()
    if not location:
        return jsonify({"ok": False, "error": "Location name is required"}), 400

    folder_name = location_to_folder_name(location)
    if not folder_name:
        return jsonify({"ok": False, "error": "Invalid location name"}), 400

    # Support single file (audio) or multiple (audio[] or multiple "audio" keys)
    audio_files = request.files.getlist("audio")
    if not audio_files or all(f.filename == "" for f in audio_files):
        if "audio" in request.files and request.files["audio"].filename != "":
            audio_files = [request.files["audio"]]
        else:
            return jsonify({"ok": False, "error": "No audio file(s) in request"}), 400

    # Filter out empty entries
    audio_files = [f for f in audio_files if f.filename]

    folder_path = DATASET_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # When pushing to GitHub, use repo state for next index so we don't overwrite
    if GITHUB_REPO:
        next_index = get_next_index_github(folder_name)
    else:
        next_index = get_next_index(folder_path, folder_name)
    saved = []
    github_ok = True
    github_err = None

    for audio_file in audio_files:
        filename = f"{folder_name}_{next_index:03d}.wav"
        output_path = folder_path / filename

        safe_name = audio_file.filename or "recording.webm"
        if not safe_name.strip():
            safe_name = "recording.webm"
        temp_path = folder_path / ("_temp_" + safe_name)
        try:
            audio_file.save(str(temp_path))
            ensure_wav_16k_mono_16bit(temp_path, output_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": str(e), "saved_so_far": saved}), 500
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

        append_metadata(filename, location)
        saved.append(filename)

        if GITHUB_REPO and output_path.exists():
            with open(output_path, "rb") as f:
                wav_bytes = f.read()
            ok1, err1 = push_file_to_github(
                f"dataset/{folder_name}/{filename}",
                wav_bytes,
                f"Add recording: {filename}",
            )
            if not ok1 and github_ok:
                github_ok = False
                github_err = err1
            elif ok1:
                ok2, err2 = push_metadata_line_to_github(filename, location)
                if not ok2 and github_ok:
                    github_ok = False
                    github_err = err2
        elif not GITHUB_REPO and github_err is None:
            github_err = "Server has no GITHUB_REPO set. Files are only on the server (lost on restart/sleep). Add GITHUB_REPO and GITHUB_TOKEN in Render env to save to your repo."

        next_index += 1

    payload = {
        "ok": True,
        "saved": saved,
        "location": location,
        "count": next_index - 1,
        "pushed_to_github": github_ok and bool(GITHUB_REPO),
    }
    if github_err:
        payload["github_error"] = github_err
    return jsonify(payload)


def check_pydub():
    """Fail fast with a clear message if pydub is not installed (e.g. wrong venv)."""
    try:
        from pydub import AudioSegment  # noqa: F401
    except ImportError:
        import sys
        print(
            "Error: pydub is not installed in this Python environment.\n"
            "With your venv activated, run:\n  pip install pydub\n"
            "You may also need ffmpeg: brew install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    check_pydub()
    ensure_dataset_structure()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", debug=debug, port=port)
