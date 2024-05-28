from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, disconnect
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import base64
import cv2
import utils
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app,  cors_allowed_origins="*")

####### Image Processing #######

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

video_data = []
full_video_data = []

def process_image(image_data):
    img_data = base64.b64decode(image_data)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            lmrks = utils.face_mesh_to_array(results, frame_rgb.shape[1], frame_rgb.shape[0])
            if lmrks is not None:
                bbox = utils.get_square_bbox(utils.get_bbox(lmrks, frame_rgb.shape[1], frame_rgb.shape[0]), frame_rgb.shape[1], frame_rgb.shape[0])
                x1, y1, x2, y2 = bbox
                cropped = frame_rgb[y1:y2, x1:x2]
                return cropped 
    return None

# 밴드패스 필터 설정
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def extract_green_channel_signal(video_frames):
    green_channel_means = [frame[:, :, 1].mean() for frame in video_frames]
    smoothed_wave = gaussian_filter(green_channel_means, sigma=2)
    filtered_signal = butter_bandpass_filter(smoothed_wave, 0.67, 3.33, fps = 30, order=2)
    return np.gradient(smoothed_wave)

def calculate_hr(state, video_frames, fps=30, window_size=150, step_size=30):
    rPPG_signal = extract_green_channel_signal(video_frames)
    bpm_per_frame = []

    for start in range(0, len(rPPG_signal) - window_size, step_size):
        end = start + window_size
        segment = rPPG_signal[start:end]
        peaks, _ = find_peaks(segment, distance=10)
        if len(peaks) > 1:
            ibi = np.diff(peaks) / fps
            bpm = 60 / np.mean(ibi) if len(ibi) > 0 else np.nan
        else:
            bpm = np.nan
        bpm_per_frame.append(bpm)

    if state == "end": return bpm_per_frame
    return bpm_per_frame[-1] if bpm_per_frame else None

# Stress Index
def calculate_SI(signal, fps, window_size=150, step_size=30):
    # 1. rPPG 신호 추출
    rPPG_Signal = extract_green_channel_signal(signal)
    rppg = rPPG_Signal - np.mean(rPPG_Signal)  # DC 컴포넌트 제거

    # 2. PPI (Peak-to-Peak Interval) 추출
    peaks, _ = find_peaks(rppg, distance=fps//2)  # 예: 20은 최소 간격 (ms)입니다.
    nn_intervals = np.diff(peaks) / fps  # PPI를 초 단위로 변환

    # 3. Mode (Mo) 및 Amplitude of Mode (AMo) 계산
    bin_width = 0.05  # 50ms를 초 단위로 변환
    hist, bin_edges = np.histogram(nn_intervals, bins=np.arange(0, np.max(nn_intervals), bin_width))
    print(f"hist : {hist}")
    print(f"bin: {bin_edges}")
    Mo = bin_edges[np.argmax(hist)]
    AMo = (np.sum((nn_intervals >= Mo - bin_width/2) & (nn_intervals < Mo + bin_width/2)) / len(nn_intervals))

    # 4. Variation range (MxDMn) 계산
    Mx = np.max(nn_intervals)
    Mn = np.min(nn_intervals)   
    MxDMn = Mx - Mn

    # 5. Baevsky stress index (SI) 계산 0~40 사이 값으로 나옴
    SI = (AMo * 100) / (2 * Mo * MxDMn)
    return SI

@app.route('/')
def hello_world():
    return "FaceMind Flask Server"

@app.route('/chat')
def chatting():
    return render_template('client.html')

# connection open(start)
@socketio.on('message')
def handle_message(json):
    print("start : ", json)
    images = json.get('images', [])
    processed_images = [process_image(image) for image in images]
    processed_images = [img for img in processed_images if img is not None]

    if processed_images:
        video_array = np.array(processed_images)
        heart_rate = calculate_hr("start", video_array)
        full_video_data.extend(processed_images)
        print('calculated hr: ', heart_rate)
        result = {'heartrate': heart_rate}
    else:
        result = {'heartrate': None}
    socketio.send(result) # client에게 메세지 전송
    socketio.sleep(0.5)

# connection 도중에 image 보내기
def example_send(json):
    print("ing : ", json)
    images = json.get('images', [])
    processed_images = [process_image(image) for image in images]
    processed_images = [img for img in processed_images if img is not None]
    
    if processed_images:
        video_array = np.array(processed_images)
        heart_rate = calculate_hr("ing", video_array)
        full_video_data.extend(processed_images)
        print('calculated hr: ', heart_rate)
        result = {'heartrate': heart_rate}
    else:
        result = {'heartrate': None}
    socketio.send(result) # client에게 메세지 전송
    socketio.sleep(0.5)

# connection 닫기 전에 최종 결과 반환한 후 connection 종료
@socketio.on('closeSending')
def disconnect_socket(json):
    print("close :", json)
    images = json.get('images', [])
    processed_images = [process_image(image) for image in images]
    processed_images = [img for img in processed_images if img is not None]

    if processed_images:
        video_array = np.array(processed_images)
        full_video_data.extend(processed_images)
        heart_rate = calculate_hr("end", full_video_data)
        print('calculated hr: ', heart_rate)
        min_hr = min(heart_rate)
        max_hr = max(heart_rate)
        mean_hr = np.mean(heart_rate)
        stress_index = calculate_SI(np.array(full_video_data), fps=30)
        result = {
            "min_hr": min_hr,
            "max_hr": max_hr,
            "mean_hr": mean_hr,
            "stress_index": stress_index
        }
    else:
        result = {
            "min_hr": None,
            "max_hr": None,
            "mean_hr": None,
            "stress_index": None
        }
        
    socketio.send(result)
    socketio.sleep(0.5)
    disconnect() # 연결 해제

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)