import cv2
import time
import os
import numpy as np
from picamera2 import Picamera2

def main():
    print("Initializing Camera...")
    
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
        print("Camera started! Press 'q' to quit.")
    except Exception as e:
        print(f"Error starting camera: {e}")
        return

    cascade_path = 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_path):
        print(f"Error: File '{cascade_path}' not found.")
        print("Please run the wget command first.")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Could not load face cascade XML file.")
        return

    while True:
        try:
            frame = picam2.capture_array()
        except Exception:
            time.sleep(0.01)
            continue
        
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        status_screen = np.zeros((480, 640, 3), dtype=np.uint8)

        if len(faces) > 0:
            status_screen[:] = (0, 255, 0)
            display_text = "UNLOCKED"
            text_color = (0, 0, 0)
            print(f"Face detected! Count: {len(faces)}")
        else:
            status_screen[:] = (0, 0, 255)
            display_text = "LOCKED"
            text_color = (255, 255, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(status_screen, display_text, (text_x, text_y), font, font_scale, text_color, thickness)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Status Screen', status_screen)
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()