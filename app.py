from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import pickle
import mediapipe as mp
import re

app = Flask(__name__)

# Allow CORS only for your frontend domain
CORS(app, resources={r"/*": {"origins": "https://sign-sense-india-ai-95-jhmk.vercel.app"}})

# Load model and initialize MediaPipe Hands
model_path = 'sign_model/sign101/model.p'  # Adjust as needed
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ready', 'success': True})

@app.route('/detect', methods=['GET'])
def check_detect():
    return jsonify({'status': 'ready', 'success': True})

@app.route('/detect', methods=['POST'])
def detect_sign():
    try:
        content = request.json
        image_data = content['image']
        base64_data = re.sub('^data:image/.+;base64,', '', image_data)

        img_bytes = base64.b64decode(base64_data)
        img_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_, data_aux = [], [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) == 42:
                    prediction = model.predict([np.asarray(data_aux)])[0]
                    probabilities = model.predict_proba([np.asarray(data_aux)])[0]
                    confidence = float(probabilities[model.classes_.tolist().index(prediction)])
                    predicted_character = prediction.upper()

                    return jsonify({
                        'sign': predicted_character,
                        'confidence': confidence,
                        'success': True
                    })

        return jsonify({'success': False, 'message': 'No hand detected'})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
