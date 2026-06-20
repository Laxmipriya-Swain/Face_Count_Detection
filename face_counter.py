import cv2
import mediapipe as mp
import json

class FaceCounter:
    def __init__(self, min_detection_confidence=0.5, model_selection=1):
        """
        model_selection:
            0 -> best for faces within 2 meters (webcam close-range)
            1 -> best for faces within 5 meters (covers background screens/posters at distance)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, frame):
        """
        Runs detection on a single BGR frame (as read by OpenCV).
        Returns: (face_count, list_of_bounding_boxes, annotated_frame)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)

        boxes = []
        annotated = frame.copy()
        h, w, _ = frame.shape

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Clamp to frame bounds (avoid negative/overflow boxes)
                x, y = max(0, x), max(0, y)
                bw, bh = min(w - x, bw), min(h - y, bh)

                confidence = detection.score[0]
                boxes.append({
                    "x": x, "y": y, "width": bw, "height": bh,
                    "confidence": round(float(confidence), 3)
                })

                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(annotated, f"{confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        face_count = len(boxes)
        cv2.putText(annotated, f"Face Count: {face_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return face_count, boxes, annotated

    def get_json_output(self, face_count):
        return json.dumps({"face_count": face_count})

    def close(self):
        self.face_detector.close()