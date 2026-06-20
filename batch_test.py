import cv2
import os
import json
from face_counter import FaceCounter

def run_batch_test(input_dir="test_images", output_dir="test_results"):
    counter = FaceCounter(min_detection_confidence=0.6, model_selection=1)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    valid_ext = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_ext):
            continue

        path = os.path.join(input_dir, filename)
        frame = cv2.imread(path)
        if frame is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        face_count, boxes, annotated = counter.detect_faces(frame)

        out_path = os.path.join(output_dir, f"annotated_{filename}")
        cv2.imwrite(out_path, annotated)

        results.append({
            "image": filename,
            "face_count": face_count,
            "boxes": boxes
        })
        print(f"{filename}: face_count={face_count}")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    counter.close()
    print(f"\nDone. Annotated images + results.json saved in '{output_dir}'")

if __name__ == "__main__":
    run_batch_test()