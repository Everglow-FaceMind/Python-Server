from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import mediapipe as mp
from utils import face_mesh_to_array, get_bbox, get_square_bbox
import base64

app = FastAPI()

# Initialize MediaPipe face mesh.
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

def derivative(signal):
    return np.gradient(signal)

def extract_green_channel_signal(video_data):
    green_channel = video_data[:, :, 1]
    green_channel_mean = green_channel.mean(axis=(0, 1))
    smoothed_wave = gaussian_filter(green_channel_mean, sigma=2)
    diff_smoothed_wave = derivative(smoothed_wave)
    return diff_smoothed_wave

def calculate_hr(video_data, fps=30, window_size=150, step_size=30):
    rPPG_Signal = extract_green_channel_signal(video_data)
    bpm_per_frame = []
    times = []

    for start in range(0, len(rPPG_Signal) - window_size, step_size):
        end = start + window_size
        segment = rPPG_Signal[start:end]
        peaks, _ = find_peaks(segment, distance=10, height=None)

        if len(peaks) > 1:
            ibi = np.diff(peaks) / fps
            bpm = 60 / np.mean(ibi) if len(ibi) > 0 else np.nan
        else:
            bpm = np.nan

        bpm_per_frame.append(bpm)
        times.append((start + end) / 2 / fps)

    return bpm_per_frame[-1] if bpm_per_frame else None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    video_data = []
    try:
        while True:
            data = await websocket.receive_text()
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    lmrks = face_mesh_to_array(results, frame_rgb.shape[1], frame_rgb.shape[0])
                    if lmrks is not None:
                        bbox = get_square_bbox(get_bbox(lmrks, frame_rgb.shape[1], frame_rgb.shape[0]), frame_rgb.shape[1], frame_rgb.shape[0])
                        x1, y1, x2, y2 = bbox
                        cropped = frame_rgb[y1:y2, x1:x2]
                        video_data.append(cropped)
                        if len(video_data) >= 150:
                            heart_rate = calculate_hr(np.array(video_data))
                            print('calculated hr: ', str(heart_rate))
                            await websocket.send_text(str(heart_rate))
                            video_data.pop(0)  # Maintain a sliding window
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
