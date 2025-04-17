import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam and MediaPipe hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Loop through webcam frames
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_.clear()
            y_.clear()

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            data_aux.clear()
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0].lower()  # Just use the predicted label directly

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
