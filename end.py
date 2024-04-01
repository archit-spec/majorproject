import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

np.random.seed(0)

def preprocess_frames(frames, image_processor):
    inputs = image_processor(frames, return_tensors="pt")
    return inputs

# Initialize the video capture object
cap = cv2.VideoCapture('http://192.168.1.9:8080/video')

# Set the frame size (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

image_processor = AutoImageProcessor.from_pretrained("archit11/videomae-base-finetuned-ucfcrime-full")
model = VideoMAEForVideoClassification.from_pretrained("archit11/videomae-base-finetuned-ucfcrime-full")

frame_buffer = []
buffer_size = 16

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Add the current frame to the buffer
    frame_buffer.append(frame)

    # Check if we have enough frames for inference
    if len(frame_buffer) >= buffer_size:
        # Preprocess the frames
        inputs = preprocess_frames(frame_buffer, image_processor)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # model predicts one of the 13 ucf-crime classes
        predicted_label = logits.argmax(-1).item()
        prediction_text = model.config.id2label[predicted_label]
        print(prediction_text)

        # Clear the frame buffer and continue from the next frame
        frame_buffer.clear()

        # Display the prediction on the frame
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
