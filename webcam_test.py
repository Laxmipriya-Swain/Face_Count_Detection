import cv2
from face_counter import FaceCounter

def run_webcam():
    counter = FaceCounter(min_detection_confidence=0.6, model_selection=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        face_count, boxes, annotated = counter.detect_faces(frame)
        print(counter.get_json_output(face_count))

        cv2.imshow("Face Count Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    counter.close()

if __name__ == "__main__":
    run_webcam()