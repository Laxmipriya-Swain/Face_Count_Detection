# Face Count Detection

A real-time face detection and counting module built for the Monitoring phase of an AI-proctoring platform. Detects and counts all faces visible in a webcam frame or static image, with handling for edge cases like reflections, posters, and background screens.

## Features
- Real-time face detection using webcam
- Batch testing on static images
- Outputs structured JSON: {"face_count": N}
- Temporal consistency filtering to reduce flicker/false positives
- Annotated output images with bounding boxes and confidence scores

## Tech Stack
- Python
- OpenCV
- MediaPipe (Face Detection)
- NumPy

## Project Structure
face-count-detection/
- face_counter.py -> Core detection logic (FaceCounter, TemporalFaceCounter)
- webcam_test.py -> Live webcam face counting
- batch_test.py -> Batch testing on test_images/
- test_images/ -> Sample test images
- test_results/ -> Annotated outputs + results.json
- requirements.txt -> Dependencies

## Setup
pip install -r requirements.txt

## Usage
Live webcam: python webcam_test.py (press q to quit)
Batch test on images: python batch_test.py (outputs annotated images and results.json to test_results/)

## Output Example
{"face_count": 2}

## Edge Cases Handled
- Reflections (mirrors/glass)
- Posters and printed photos
- Background screens showing faces

## Author
Laxmipriya Swain
