import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue  # skip files

    for img_name in os.listdir(label_path):
        data_aux = []
        x_ = []
        y_ = []

        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Skipped invalid image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                # Only add data if it has exactly 42 values (21 landmarks × 2 coords)
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(label)
                else:
                    print(f"[WARNING] Skipped image due to incomplete landmarks: {img_path}")
        else:
            print(f"[INFO] No hand detected in: {img_path}")

# Save to pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"[DONE] Saved {len(data)} valid samples from {DATA_DIR}")
