from flask import Flask, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
try:
    with open('./model.pkl', 'rb') as f:  # Ensure the model file is correctly named
        model_dict = pickle.load(f)
        model = model_dict['model']
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'Time',
    1: 'Fork',
    2: 'Coat',
    3: 'Sorry',
    4: 'Good',
    5: 'Love',
    6: 'Request',
    7: 'Thank You',
    8: 'Help',
    9: 'What',
    10: 'Stop',
    11: 'Hungry',
    12: 'Stand',
    13: 'Hello',
    14: 'Today'
}

@app.route('/')
def index():
    return "Hello, this is the root page of the ASL app."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    try:
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        frame = np.array(image)

        if frame.ndim != 3 or frame.shape[2] != 3:
            return jsonify({"error": "Invalid image format"}), 400

        data_aux = []
        left_x, left_y = [], []
        right_x, right_y = [], []

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    if i < 21:  # Left hand keypoints
                        left_x.append(landmark.x)
                        left_y.append(landmark.y)
                    else:  # Right hand keypoints
                        right_x.append(landmark.x)
                        right_y.append(landmark.y)

            # Append keypoints of both hands to data_aux
            data_aux.extend(left_x)
            data_aux.extend(left_y)
            data_aux.extend(right_x)
            data_aux.extend(right_y)

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
            except Exception as e:
                predicted_character = "Error during prediction: " + str(e)
        else:
            predicted_character = "No hand landmarks detected"

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"prediction": predicted_character})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
