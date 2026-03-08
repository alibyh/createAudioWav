/**
 * Place Names Voice Recorder – frontend
 * Uses MediaRecorder API; uploads audio to Flask backend for WAV conversion and storage.
 */

(function () {
    "use strict";

    const locationInput = document.getElementById("location-input");
    const currentLocationEl = document.getElementById("current-location");
    const recordingStatusEl = document.getElementById("recording-status");
    const recordingCountEl = document.getElementById("recording-count");
    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");
    const btnSave = document.getElementById("btn-save");
    const messageEl = document.getElementById("message");
    const pendingCountEl = document.getElementById("pending-count");

    const MAX_PENDING = 5;

    let mediaRecorder = null;
    let recordedChunks = [];
    let stream = null;
    /** @type {{ blob: Blob; mimeType: string }[]} */
    let pendingRecordings = [];

    function showMessage(text, type) {
        messageEl.textContent = text;
        messageEl.className = "message " + (type || "");
    }

    function clearMessage() {
        messageEl.textContent = "";
        messageEl.className = "message";
    }

    function setStatusIdle() {
        recordingStatusEl.textContent = "Idle";
        recordingStatusEl.className = "status-value status-idle";
    }

    function setStatusRecording() {
        recordingStatusEl.textContent = "Recording";
        recordingStatusEl.className = "status-value status-recording";
    }

    function locationToFolderName(location) {
        return location.trim().replace(/\s+/g, "_") || "";
    }

    function updateCurrentLocation() {
        const loc = locationInput.value.trim();
        currentLocationEl.textContent = loc || "—";
    }

    function updateRecordingCount() {
        const folder = locationToFolderName(locationInput.value);
        if (!folder) {
            recordingCountEl.textContent = "0";
            return;
        }
        fetch("/api/recordings/count?location_folder=" + encodeURIComponent(folder))
            .then(function (res) {
                return res.json();
            })
            .then(function (data) {
                recordingCountEl.textContent = String(data.count || 0);
            })
            .catch(function () {
                recordingCountEl.textContent = "—";
            });
    }

    function updatePendingUI() {
        const n = pendingRecordings.length;
        pendingCountEl.textContent = n + " / " + MAX_PENDING;
        btnSave.disabled = n === 0;
        btnSave.textContent = n === 0 ? "Save All" : "Save All (" + n + ")";
        btnStart.disabled = n >= MAX_PENDING;
    }

    locationInput.addEventListener("input", function () {
        updateCurrentLocation();
        updateRecordingCount();
    });
    locationInput.addEventListener("blur", updateRecordingCount);

    function requestMicrophone() {
        return navigator.mediaDevices.getUserMedia({ audio: true });
    }

    function startRecording() {
        const location = locationInput.value.trim();
        if (!location) {
            showMessage("Enter a location name before recording.", "error");
            return;
        }
        if (pendingRecordings.length >= MAX_PENDING) {
            showMessage("Maximum " + MAX_PENDING + " recordings. Save All first.", "error");
            return;
        }

        clearMessage();
        recordedChunks = [];

        requestMicrophone()
            .then(function (audioStream) {
                stream = audioStream;
                const options = { mimeType: "audio/webm;codecs=opus", audioBitsPerSecond: 128000 };
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = "audio/webm";
                }
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = "";
                }
                mediaRecorder = new MediaRecorder(audioStream, options);
                mediaRecorder.ondataavailable = function (e) {
                    if (e.data.size > 0) recordedChunks.push(e.data);
                };
                mediaRecorder.start(100);
                setStatusRecording();
                btnStart.disabled = true;
                btnStop.disabled = false;
            })
            .catch(function (err) {
                showMessage("Microphone access denied or failed: " + (err.message || "unknown error"), "error");
            });
    }

    function stopRecording() {
        if (!mediaRecorder || mediaRecorder.state === "inactive") return;
        mediaRecorder.stop();
        if (stream) {
            stream.getTracks().forEach(function (t) {
                t.stop();
            });
            stream = null;
        }
        setStatusIdle();
        btnStart.disabled = false;
        btnStop.disabled = true;

        if (recordedChunks.length > 0 && pendingRecordings.length < MAX_PENDING) {
            const blob = new Blob(recordedChunks, { type: mediaRecorder?.mimeType || "audio/webm" });
            pendingRecordings.push({ blob: blob, mimeType: mediaRecorder?.mimeType || "audio/webm" });
            updatePendingUI();
        }
    }

    function saveRecording() {
        const location = locationInput.value.trim();
        if (!location) {
            showMessage("Enter a location name before saving.", "error");
            return;
        }
        if (pendingRecordings.length === 0) {
            showMessage("No recordings to save. Record at least one clip, then Save All.", "error");
            return;
        }

        clearMessage();
        showMessage("Saving " + pendingRecordings.length + " recording(s)…", "info");

        const formData = new FormData();
        formData.append("location", location);
        pendingRecordings.forEach(function (rec, i) {
            formData.append("audio", rec.blob, "recording_" + i + ".webm");
        });

        fetch("/api/upload", {
            method: "POST",
            body: formData,
        })
            .then(function (res) {
                return res.json().then(function (data) {
                    return { ok: res.ok, status: res.status, data: data };
                });
            })
            .then(function (result) {
                if (result.ok && result.data.ok) {
                    const saved = result.data.saved || [];
                    pendingRecordings = [];
                    updatePendingUI();
                    updateRecordingCount();
                    showMessage("Saved " + saved.length + " file(s): " + (saved.join(", ")) + ".", "success");
                } else {
                    showMessage(result.data.error || "Upload failed.", "error");
                }
            })
            .catch(function (err) {
                showMessage("Upload failed: " + (err.message || "network error"), "error");
            });
    }

    btnStart.addEventListener("click", startRecording);
    btnStop.addEventListener("click", stopRecording);
    btnSave.addEventListener("click", saveRecording);

    updateCurrentLocation();
    updateRecordingCount();
    updatePendingUI();
})();
