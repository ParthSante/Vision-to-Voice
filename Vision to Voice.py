#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# prompt: mount drive

from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('unzip /content/drive/MyDrive/Gohan_Dataset/RT3_Dataset/train.zip -d /content/RT3_Dataset/')


# In[ ]:


get_ipython().system('unzip /content/drive/MyDrive/Gohan_Dataset/RT3_Dataset/valid.zip -d /content/RT3_Dataset/')


# In[ ]:


get_ipython().system('pip install ultralytics')


# In[ ]:


from ultralytics import YOLO
model = YOLO("yolov8n.pt")


# In[ ]:


# Train the model
model.train(data='/content/drive/MyDrive/Gohan_Dataset/RT3_Dataset/data.yaml' , epochs=4, batch=24, imgsz=640, name="yolov8_custom")


# In[ ]:





# In[ ]:





# # Text to speech

# In[ ]:


get_ipython().system('pip install ultralytics')

get_ipython().system('pip install gTTS')

from gtts import gTTS


# In[ ]:


from ultralytics import YOLO
from gtts import gTTS
from IPython.display import Audio, display
from collections import defaultdict
import cv2
from google.colab.patches import cv2_imshow  # Use cv2_imshow in Colab

# Load the custom-trained YOLO model
model = YOLO('/content/drive/MyDrive/yolov8_custom/weights/best.pt')  # Update to your model path

def detect_objects(image_path, confidence=0.5):
    """Detect objects in the image and return results."""
    try:
        results = model.predict(source=image_path, save=True, conf=confidence)
        if not results:
            print("No objects detected.")
        return results
    except Exception as e:
        print(f"Error during object detection: {e}")
        return None

def count_objects(results):
    """Count detected objects by class."""
    detected_classes = results[0].names  # Get class names
    counts = defaultdict(int)  # Dictionary to hold object counts

    # Count each detected object
    for box in results[0].boxes:
        cls = int(box.cls)
        class_name = detected_classes[cls]
        counts[class_name] += 1

    return counts

def generate_count_feedback(counts):
    """Generate text summary of object counts and convert it to speech."""
    # Generate the feedback text in the format: 'Detected objects are 3 person, 2 cars, 1 bike'
    feedback_texts = [f"{count} {name}" for name, count in counts.items()]
    combined_feedback = "Detected objects are " + ", ".join(feedback_texts) + "."

    # Convert to audio using gTTS
    tts = gTTS(text=combined_feedback, lang='en', slow=False)
    tts.save("count_feedback.wav")
    return Audio("count_feedback.wav", autoplay=True)

# Example usage
image_path = '/content/gettyimages-517928072-1024x1024.jpg'  # Replace with your image path
results = detect_objects(image_path)

if results:
    # Show the image with bounding boxes
    cv2_imshow(results[0].plot())  # This will display the image with bounding boxes

    # Count detected objects and generate feedback
    counts = count_objects(results)
    audio_output = generate_count_feedback(counts)
    display(audio_output)
else:
    print("No results to process.")


# In[ ]:


from ultralytics import YOLO

# Load your custom-trained YOLO model
model = YOLO('/content/drive/MyDrive/yolov8_custom/weights/best.pt')  # Change this to 'last.pt' if needed

# Predict objects in an image
results = model.predict(source='/content/depositphotos_356489422-stock-photo-new-delhi-india-nobember-2019.jpg', save=True, conf=0.30)  # Adjust confidence threshold as needed

# Display results
results[0].show()


# In[ ]:


from ultralytics import YOLO
from gtts import gTTS
from IPython.display import Audio, display
import cv2

# Load your custom-trained YOLO model
model = YOLO('/content/drive/MyDrive/yolov8_custom/weights/best.pt')  # Adjust path to your model

# Define object size estimates and class category mapping
object_size_estimates = {
    'person': 1.7, 'animal': 1.0, 'small_object': 0.2, 'medium_object': 0.5,
    'large_object': 1.5, 'vehicle': 1.5, 'large_vehicle': 3.0
}

class_category_map = {
    'person': 'person', 'dog': 'animal', 'cat': 'animal', 'elephant': 'animal', 'giraffe': 'animal',
    'bottle': 'small_object', 'cup': 'small_object', 'remote': 'small_object', 'cell phone': 'small_object',
    'laptop': 'medium_object', 'keyboard': 'medium_object', 'book': 'small_object', 'pottedplant': 'medium_object',
    'bicycle': 'vehicle', 'motorbike': 'vehicle', 'car': 'vehicle', 'bus': 'large_vehicle', 'train': 'large_vehicle',
    'truck': 'large_vehicle', 'aeroplane': 'large_vehicle', 'boat': 'large_vehicle'
}

def calculate_distance(real_size, focal_length, bbox_dimension):
    return (real_size * focal_length) / bbox_dimension

# Set camera focal length
focal_length = 500  # Approximate focal length in pixels

def detect_objects(image_path):
    try:
        # Perform object detection
        results = model.predict(source=image_path, save=True, conf=0.5)  # Adjust confidence threshold as needed

        # Check if results are obtained
        if results:
            # Display the image with bounding boxes
            results[0].show()  # Shows the first result with bounding boxes
        else:
            print("No objects detected.")

        return results

    except Exception as e:
        print(f"Error during object detection: {e}")
        return None

def process_distances(results):
    detected_classes = results[0].names  # Class names from YOLO results
    detected_distances = []

    # Iterate over detected objects and calculate distances
    for box in results[0].boxes:
        cls = int(box.cls)  # Class index
        class_name = detected_classes[cls]  # Get the name of the detected class

        # Get category and size estimate for the object based on the class
        category = class_category_map.get(class_name, 'medium_object')
        real_size = object_size_estimates[category]

        # Get bounding box coordinates in xyxy format
        x1, y1, x2, y2 = box.xyxy[0]

        # Calculate bbox height and width in pixels
        bbox_height = (y2 - y1).item()
        bbox_width = (x2 - x1).item()

        # Determine dimension to use for distance calculation based on object category
        if category in ['person', 'animal']:  # For tall objects, use height
            distance = calculate_distance(real_size, focal_length, bbox_height)
        else:  # For wide objects (vehicles, etc.), use width
            distance = calculate_distance(real_size, focal_length, bbox_width)

        # Store detected object information with calculated distance and position
        detected_distances.append((class_name, distance, (x1, y1, x2, y2)))

    # Sort detected objects by distance
    detected_distances.sort(key=lambda x: x[1])  # Sort by distance in ascending order

    # Extract nearest, medium, and farthest objects
    nearest_object = detected_distances[0] if detected_distances else None
    farthest_object = detected_distances[-1] if detected_distances else None
    medium_index = len(detected_distances) // 2
    medium_object = detected_distances[medium_index] if detected_distances else None

    return nearest_object, medium_object, farthest_object, detected_distances

def generate_audio_feedback(nearest_object, medium_object, farthest_object):
    feedback_texts = []

    def get_direction(x_center):
        """Determine direction based on x-center relative to image center (320 for 640px width)."""
        if x_center < 320:
            return "left"
        elif x_center > 320:
            return "right"
        return "center"

    # Get direction and feedback for the nearest object
    if nearest_object:
        feedback_texts.append(f"The nearest object is a {nearest_object[0]} at {nearest_object[1]:.2f} meters.")

        # Determine direction to nearest object based on bounding box coordinates
        x1, y1, x2, y2 = nearest_object[2]
        nearest_direction = get_direction((x1 + x2) / 2)
        feedback_texts.append(f"It is to your {nearest_direction}.")
    else:
        feedback_texts.append("No nearest object detected.")

    # Get direction and feedback for the farthest object
    if farthest_object:
        feedback_texts.append(f"The farthest object is a {farthest_object[0]} at {farthest_object[1]:.2f} meters.")

        # Determine direction to farthest object based on bounding box coordinates
        x1, y1, x2, y2 = farthest_object[2]
        farthest_direction = get_direction((x1 + x2) / 2)
        feedback_texts.append(f"It is to your {farthest_direction}.")
    else:
        feedback_texts.append("No farthest object detected.")

    # Safety direction suggestion
    if nearest_object and nearest_direction == "left":
        feedback_texts.append("It is safer to move to the right.")
    elif nearest_object and nearest_direction == "right":
        feedback_texts.append("It is safer to move to the left.")
    else:
        feedback_texts.append("You are facing the object, proceed with caution.")

    # Add medium object feedback
    if medium_object:
        feedback_texts.append(f"The middle object is a {medium_object[0]} at {medium_object[1]:.2f} meters. Be aware of that.")
    else:
        feedback_texts.append("No middle object detected.")

    # Combine the feedback texts
    combined_feedback = " ".join(feedback_texts)

    # Convert text to speech using gTTS
    gtts_object = gTTS(text=combined_feedback, lang='en', slow=False)
    gtts_object.save("detection_feedback.wav")

    # Play the audio
    return Audio("detection_feedback.wav", autoplay=True)

# Example usage
image_path = '/content/-1x-1.webp'  # Replace with your test image path
results = detect_objects(image_path)  # Detect objects and show image

if results:  # Ensure results are obtained before processing distances
    nearest_object, medium_object, farthest_object, detected_distances = process_distances(results)  # Process distances
    audio_output = generate_audio_feedback(nearest_object, medium_object, farthest_object)  # Generate audio feedback

    # Display the audio output
    display(audio_output)
else:
    print("No results to process.")


# In[ ]:





# In[ ]:


from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow  # Use cv2_imshow for Colab compatibility

# Load your custom-trained YOLO model
model = YOLO('/content/drive/MyDrive/yolov8_custom/weights/best.pt')  # Adjust path to your model

# Define object size estimates and class category mapping
object_size_estimates = {
    'person': 1.7, 'animal': 1.0, 'small_object': 0.2, 'medium_object': 0.5,
    'large_object': 1.5, 'vehicle': 1.5, 'large_vehicle': 3.0
}

class_category_map = {
    'person': 'person', 'dog': 'animal', 'cat': 'animal', 'elephant': 'animal', 'giraffe': 'animal',
    'bottle': 'small_object', 'cup': 'small_object', 'remote': 'small_object', 'cell phone': 'small_object',
    'laptop': 'medium_object', 'keyboard': 'medium_object', 'book': 'small_object', 'pottedplant': 'medium_object',
    'bicycle': 'vehicle', 'motorbike': 'vehicle', 'car': 'vehicle', 'bus': 'large_vehicle', 'train': 'large_vehicle',
    'truck': 'large_vehicle', 'aeroplane': 'large_vehicle', 'boat': 'large_vehicle'
}

def calculate_distance(real_size, focal_length, bbox_dimension):
    return (real_size * focal_length) / bbox_dimension

# Set camera focal length
focal_length = 500  # Approximate focal length in pixels

def process_frame(frame):
    """Process each frame to detect objects and annotate them with bounding boxes and distances."""
    results = model.predict(source=frame, conf=0.5)
    detected_classes = results[0].names

    for box in results[0].boxes:
        cls = int(box.cls)
        class_name = detected_classes[cls]

        # Get size estimate for the object class
        category = class_category_map.get(class_name, 'medium_object')
        real_size = object_size_estimates[category]

        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        bbox_height = (y2 - y1).item()
        bbox_width = (x2 - x1).item()

        # Calculate distance based on object dimensions
        if category in ['person', 'animal']:
            distance = calculate_distance(real_size, focal_length, bbox_height)
        else:
            distance = calculate_distance(real_size, focal_length, bbox_width)

        # Annotate frame with bounding box and distance label
        label = f"{class_name} {distance:.2f}m"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def process_video(input_path, output_path):
    """Process the input video, annotate frames, and save the output."""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define video codec and writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame.")
                break

            # Process the frame and annotate it
            annotated_frame = process_frame(frame)

            # Write the processed frame to the output video
            out.write(annotated_frame)

            # Optional: Display the frame in Colab for real-time preview
            cv2_imshow(annotated_frame)

            # To exit loop by pressing 'q' in a non-Colab environment, uncomment below
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Path to input and output video files
input_video_path = '/content/WhatsApp Video 2024-11-09 at 10.59.24 PM.mp4'  # Path to your input video
output_video_path = '/content/output_video_with_bboxes.mp4'  # Path where output video will be saved

# Process and save the video
process_video(input_video_path, output_video_path)
print("Processing complete. Output video saved.")


# In[ ]:





# # RUN THIS FOLLOWING CODE ON JUPYTER NOTEBOOK AS COLAB DOESN'T SUPPORTS HARDWARE

# In[ ]:


import os

# Set the specified path as the current working directory
os.chdir("C:\\Users\\Desktop\\Vision_To_Voice\\YOLOv8nano_TFLite_model")

# Verify the current working directory
print("Current Working Directory:", os.getcwd())


# In[ ]:


from ultralytics import YOLO
import cv2

# Load your custom-trained YOLO model
# Replace your best.pt file according to your current wroking directory
model = YOLO(r"C:\Users\Desktop\Vision_To_Voice\YOLOv8nano_TFLite_model\runs\detect\yolov8_custom\weights\best.pt")

# Define object size estimates and class category mapping
object_size_estimates = {
    'person': 1.7, 'animal': 1.0, 'small_object': 0.2, 'medium_object': 0.5,
    'large_object': 1.5, 'vehicle': 1.5, 'large_vehicle': 3.0
}

class_category_map = {
    'person': 'person', 'dog': 'animal', 'cat': 'animal', 'elephant': 'animal', 'giraffe': 'animal',
    'bottle': 'small_object', 'cup': 'small_object', 'remote': 'small_object', 'cell phone': 'small_object',
    'laptop': 'medium_object', 'keyboard': 'medium_object', 'book': 'small_object', 'pottedplant': 'medium_object',
    'bicycle': 'vehicle', 'motorbike': 'vehicle', 'car': 'vehicle', 'bus': 'large_vehicle', 'train': 'large_vehicle',
    'truck': 'large_vehicle', 'aeroplane': 'large_vehicle', 'boat': 'large_vehicle'
}

def calculate_distance(real_size, focal_length, bbox_dimension):
    return (real_size * focal_length) / bbox_dimension

# Set camera focal length
focal_length = 500  # Approximate focal length in pixels

def process_frame(frame):
    """Detect and annotate objects in a single frame with bounding boxes and distance labels."""
    results = model.predict(source=frame, conf=0.5)  # Adjust confidence threshold as needed
    detected_classes = results[0].names

    for box in results[0].boxes:
        cls = int(box.cls)
        class_name = detected_classes[cls]

        # Get size estimate for the object class
        category = class_category_map.get(class_name, 'medium_object')
        real_size = object_size_estimates[category]

        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        bbox_height = (y2 - y1).item()
        bbox_width = (x2 - x1).item()

        # Calculate distance based on object dimensions
        if category in ['person', 'animal']:
            distance = calculate_distance(real_size, focal_length, bbox_height)
        else:
            distance = calculate_distance(real_size, focal_length, bbox_width)

        # Annotate frame with bounding box and distance label
        label = f"{class_name} {distance:.2f}m"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def live_object_detection():
    """Perform real-time object detection using webcam."""
    cap = cv2.VideoCapture(0)  # 0 for the default camera; use 1 or other indices for external webcams

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Process the frame for object detection and distance annotation
            annotated_frame = process_frame(frame)

            # Display the frame with annotations
            cv2.imshow("Live Object Detection", annotated_frame)

            # Press 'q' to quit the live stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Run live object detection
live_object_detection()


# In[ ]:





# In[ ]:




