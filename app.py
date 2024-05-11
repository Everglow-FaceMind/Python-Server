from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, disconnect
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app,  cors_allowed_origins="*")

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
    image = json.get('image', '')
    # 📌 심박수 계산하는 함수 작성 (dictionary 형식으로 return 해주세요!)
    #ex. result = calculate("start", image)
    result = image
    socketio.send(result) # client에게 메세지 전송
    socketio.sleep(0.5)

# connection 도중에 image 보내기
@socketio.on('ingSending')
def example_send(json):
    print("ing : ", json)
    image = json.get('image', '')
    # 📌 심박수 계산하는 함수 작성 (dictionary 형식으로 return 해주세요!)
    #ex. result = calculate("ing", image)
    result = image
    socketio.send(result) # client에게 메세지 전송
    socketio.sleep(0.5)

# connection 닫기 전에 최종 결과 반환한 후 connection 종료
@socketio.on('closeSending')
def disconnect_socket(json):
    print("close :", json)
    image = json.get('image', '')
    # 📌 최종결과 계산하는 함수 작성 (dictionary 형식으로 return 해주세요!)
    #ex. result = calculate("end", image)
    result = image
    socketio.send(result)
    socketio.sleep(0.5)
    disconnect() # 연결 해제

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)