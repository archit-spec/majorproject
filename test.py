import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
import threading

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
previous_labels = []
top_confidences = []  # Initialize top_confidences

# Create a window and set its size
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600)  # Adjust the size as needed

def update_prediction(frame, top_labels, top_confidences):
    for i, (label, confidence) in enumerate(zip(top_labels, top_confidences)):
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('Video', frame)

def inference_thread():
    global previous_labels, top_confidences

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

            # Get the top 3 predicted labels and their confidence scores
            top_k = 3
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            top_labels = [model.config.id2label[idx.item()] for idx in top_indices[0]]
            top_confidences = top_probs[0].tolist()  # Update top_confidences

            # Check if the predicted labels are different from the previous labels
            if top_labels != previous_labels:
                previous_labels = top_labels
                print("Predicted class:", top_labels[0])  # Print the predicted class for debugging

            # Clear the frame buffer and continue from the next frame
            frame_buffer.clear()

            # Update the text on the frame in the main thread
            threading.Thread(target=update_prediction, args=(frame.copy(), top_labels, top_confidences)).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

# Start the inference thread
inference_thread = threading.Thread(target=inference_thread)
inference_thread.start()

# Keep the main thread alive
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
