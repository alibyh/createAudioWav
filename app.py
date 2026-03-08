"""
Flask backend for collecting labeled voice recordings of place names.
Stores WAV files in dataset/<location_folder>/ and tracks metadata in dataset/metadata.csv.
"""

import os
import csv
from pathlib import Path

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
METADATA_PATH = DATASET_DIR / "metadata.csv"
MAX_UPLOAD_MB = 10


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recordings/count", methods=["GET"])
def get_recording_count():
    """Return count of recordings for a location (folder name with underscores)."""
    folder_name = request.args.get("location_folder", "").strip()
    if not folder_name:
        return jsonify({"count": 0})
    folder_path = DATASET_DIR / folder_name
    if not folder_path.is_dir():
        return jsonify({"count": 0})
    count = len(list(folder_path.glob("*.wav")))
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

    next_index = get_next_index(folder_path, folder_name)
    saved = []

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
        next_index += 1

    return jsonify({
        "ok": True,
        "saved": saved,
        "location": location,
        "count": next_index - 1,
    })


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
    app.run(debug=True, port=5000)
