import cv2

# Built-in Haar cascade path from the opencv-python package
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError(f"Could not load Haar cascade at: {cascade_path}")

# 0 = default webcam; try 1 or 2 if you have multiple cameras
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("• Close Zoom/Meet/etc.")
    print("• Windows: Settings > Privacy & security > Camera > allow apps.")
    exit(1)

print("Press 'q' t0 quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection (press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
