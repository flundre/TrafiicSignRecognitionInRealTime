import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Завантаження моделі
model = load_model('traffic_sign_model.keras')
cwd = os.getcwd()

# Список класів (від '00000' до '00061')
class_names = [f"{i:05d}" for i in range(62)]

img_width, img_height = 64, 64

def get_center_square_roi(frame, size):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    half_size = size // 2

    x1 = max(cx - half_size, 0)
    y1 = max(cy - half_size, 0)
    x2 = min(cx + half_size, w)
    y2 = min(cy + half_size, h)

    roi = frame[y1:y2, x1:x2]

    roi_h, roi_w = roi.shape[:2]
    if roi_h != roi_w:
        size_min = min(roi_h, roi_w)
        roi = roi[0:size_min, 0:size_min]

    return roi

def predict_frame(frame, model, class_names):
    roi = get_center_square_roi(frame, size=200)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    roi_resized = cv2.resize(roi_rgb, (img_width, img_height))

    input_array = np.expand_dims(roi_resized, axis=0)

    pred_probs = model.predict(input_array)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    pred_class = class_names[pred_index]
    confidence = pred_probs[0][pred_index]

    print(f"[DEBUG] Top prediction: {pred_class}, confidence: {confidence:.2f}")
    return pred_class, confidence

#Запуск камери
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не вдалося відкрити камеру.")
else:
    print("Камера запущена. Натисніть 'q' для виходу.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не вдалося зчитати кадр.")
            break

        pred_class, confidence = predict_frame(frame, model, class_names)

        label = f"Prediction: {pred_class} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow('Traffic Sign Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()