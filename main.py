# main.py
import os
import uvicorn
import asyncio
import traceback
from datetime import datetime

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.preprocessing import stream_eeg_data, extract_features
from src.model_quantum import predict_brain_state, compute_raw_band_powers

app = FastAPI()

# DEV: allow everything while debugging CORS / uploads
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

SESSION_DATA = {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".edf"):
        return JSONResponse({"error": "Only EDF files are supported"}, status_code=400)

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        print(f"[upload] Saved incoming file to {file_path} ({len(contents)} bytes)")

        if not os.path.exists(file_path):
            raise RuntimeError(f"Failed to write uploaded file to {file_path}")

        try:
            feats = extract_features(file_path)
        except Exception as e:
            tb = traceback.format_exc()
            print("[upload] extract_features failed:\n", tb)
            return JSONResponse({
                "error": "Failed reading EDF",
                "traceback": tb,
                "exception": str(e)
            }, status_code=500)

        # Save sampling frequency (sfreq) and features preview (larger window)
        sfreq = feats.get("sfreq", 250.0) or 250.0
        SESSION_DATA["sfreq"] = float(sfreq)

        try:
            first_chan = feats.get("features", [])[0] if feats.get("features") is not None else None
            if first_chan is not None:
                # keep larger preview for PSD (up to 2048 samples)
                n_preview = min(first_chan.shape[0] if hasattr(first_chan, "shape") else len(first_chan), 2048)
                # convert to list
                preview_samples = (first_chan[:n_preview].tolist()
                                   if hasattr(first_chan, "tolist")
                                   else list(first_chan[:n_preview]))
                SESSION_DATA["features_preview"] = preview_samples
                print(f"[upload] preview length={len(preview_samples)} samples, sfreq={SESSION_DATA['sfreq']}")
            else:
                SESSION_DATA["features_preview"] = []
        except Exception:
            SESSION_DATA["features_preview"] = []

        # Predict brain state once using larger window (not 50)
        try:
            seed_samples = SESSION_DATA["features_preview"][:1024] if SESSION_DATA.get("features_preview") else []
            brain_state_pred = predict_brain_state(seed_samples, fs=SESSION_DATA.get("sfreq", 250.0))
        except Exception as e:
            tb = traceback.format_exc()
            print("[upload] model_quantum predict failed:\n", tb)
            brain_state_pred = "Unknown"

        SESSION_DATA["file_name"] = file.filename
        SESSION_DATA["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        SESSION_DATA["epoch_losses"] = []
        SESSION_DATA["brain_stats"] = {}
        try:
            frequency = round(feats["duration"] / len(feats["features"][0]), 2) if feats.get("features") else "N/A"
        except Exception:
            frequency = "N/A"
        SESSION_DATA["frequency"] = frequency
        SESSION_DATA["brain_state"] = brain_state_pred

        print(f"[upload] OK: {file.filename} channels={len(feats.get('channels', []))} brain_state={brain_state_pred}")

        return {"channels": feats.get("channels", []), "brain_state": "Ready"}

    except Exception as e:
        tb = traceback.format_exc()
        print("[upload] Unexpected error:\n", tb)
        return JSONResponse({"error": "Upload handler failed", "exception": str(e), "traceback": tb}, status_code=500)


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    edf_files = sorted([f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".edf")])
    if not edf_files:
        await websocket.send_json({"error": "No uploaded EDF file"})
        await websocket.close()
        return

    file_path = os.path.join(UPLOAD_DIR, edf_files[-1])
    print("[stream] Starting stream from:", file_path)

    try:
        brain_state_fixed = SESSION_DATA.get("brain_state", "Streaming")
        async for chunk in stream_eeg_data(file_path, chunk_size=50, brain_state=brain_state_fixed):
            left_right = chunk.get("left_right", [0, 0])
            metrics = chunk.get("metrics", {"focus": 0, "stress": 0, "health": 0})
            epoch_losses = chunk.get("epoch_losses", [])
            brain_state = chunk.get("brain_state", brain_state_fixed)

            SESSION_DATA["epoch_losses"] = epoch_losses
            SESSION_DATA["brain_stats"] = {
                "left": left_right[0],
                "right": left_right[1],
                "focus": metrics.get("focus", 0),
                "stress": metrics.get("stress", 0),
                "health": metrics.get("health", 0),
                "brain_state": brain_state,
            }

            payload = {
                "channels": chunk.get("channels", []),
                "signals": chunk.get("signals", []),
                "brain_state": brain_state,
                "left_right": left_right,
                "metrics": metrics,
                "epoch_losses": epoch_losses,
            }

            try:
                await websocket.send_json(payload)
            except Exception as e:
                print("[stream] websocket send error:", e)
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("[stream] Client disconnected")
    except Exception as e:
        print("[stream] Unexpected error:\n", traceback.format_exc())
    finally:
        try:
            await websocket.close()
        except:
            pass
        print("[stream] Stream ended")


@app.get("/download_pdf")
async def download_pdf():
    file_name = SESSION_DATA.get("file_name")
    if not file_name:
        raise HTTPException(status_code=400, detail="No session available. Upload an EDF first.")

    start_time = SESSION_DATA.get("start_time", "N/A")
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch_losses = SESSION_DATA.get("epoch_losses", [])
    frequency = SESSION_DATA.get("frequency", "N/A")
    stats = SESSION_DATA.get("brain_stats", {})
    brain_state = SESSION_DATA.get("brain_state", "N/A")

    safe_base = os.path.splitext(file_name)[0]
    pdf_path = os.path.join(REPORTS_DIR, f"{safe_base}_report.pdf")

    preview = SESSION_DATA.get("features_preview", [])
    fs = SESSION_DATA.get("sfreq", 250.0)
    try:
        band_powers_raw = compute_raw_band_powers(preview, fs=fs).astype(float).tolist()
    except Exception as e:
        print("[download_pdf] compute_raw_band_powers failed:", e)
        band_powers_raw = [0.0, 0.0, 0.0, 0.0]

    try:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawString(200, height - 50, "🧠 BCI Session Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 100, f"File Name: {file_name}")
        c.drawString(50, height - 120, f"Start Time: {start_time}")
        c.drawString(50, height - 140, f"End Time: {end_time}")
        c.drawString(50, height - 160, f"Frequency: {frequency}")
        c.drawString(50, height - 180, f"Brain State: {brain_state}")

        c.drawString(50, height - 210, "Epoch Losses:")
        y_losses = height - 230
        if epoch_losses:
            for i, loss in enumerate(epoch_losses):
                if i >= 10:
                    c.drawString(70, y_losses - i * 16, "... (more)"); break
                c.drawString(70, y_losses - i * 16, f"Epoch {i+1}: {loss}")
        else:
            c.drawString(70, y_losses, "No epoch losses recorded")

        c.drawString(50, y_losses - 90, "Spectral band-power (symbols):")
        y_bp = y_losses - 110
        symbols = ["Ω₁", "Ω₂", "Ω₃", "Ω₄"]
        for i, val in enumerate(band_powers_raw):
            try:
                txt = f"{symbols[i]}: {val:.4e}"
            except Exception:
                txt = f"{symbols[i]}: {val}"
            c.drawString(70, y_bp - i * 18, txt)

        y_stats = y_bp - 100
        c.drawString(50, y_stats, "Brain Stats:")
        y = y_stats - 20
        if stats:
            for key in ("left", "right", "focus", "stress", "health", "brain_state"):
                if key in stats:
                    c.drawString(70, y, f"{key.capitalize()}: {stats[key]}")
                    y -= 18
        else:
            c.drawString(70, y, "No brain stats recorded")
            y -= 18

        c.save()
    except Exception as e:
        tb = traceback.format_exc()
        print("[download_pdf] PDF creation failed:\n", tb)
        raise HTTPException(status_code=500, detail=f"Failed to create PDF: {str(e)}")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Report not found after creation.")

    return FileResponse(pdf_path, filename=os.path.basename(pdf_path))


@app.get("/session_status")
async def session_status():
    return SESSION_DATA


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
