import cv2
import face_recognition
import os
import pickle
import mediapipe as mp
from datetime import datetime

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Paths
known_faces_dir = "employee_faces"  # Directory to store employee images
encoding_file = "employee_encodings.pkl"  # File to store encoded employee data
attendance_log = "attendance_log.csv"  # File to log attendance

# Function to encode employee images
def encode_faces():
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        filepath = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(filepath)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use filename as the name
    
    # Save encodings to a file for future use
    with open(encoding_file, 'wb') as file:
        pickle.dump((known_face_encodings, known_face_names), file)

    print("Employee faces encoded and saved.")

# Load encodings from the file
def load_encodings():
    if os.path.exists(encoding_file):
        with open(encoding_file, 'rb') as file:
            return pickle.load(file)
    else:
        return [], []

# Mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Append attendance to the log
    with open(attendance_log, "a") as file:
        file.write(f"{date_str},{time_str},{name}\n")

    print(f"Attendance marked for {name} at {time_str} on {date_str}.")

# Initialize face encodings
known_face_encodings, known_face_names = load_encodings()

if not known_face_encodings:
    print("No encodings found. Encoding faces...")
    encode_faces()
    known_face_encodings, known_face_names = load_encodings()

# Initialize webcam capture
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MediaPipe
        results = face_detection.process(rgb_frame)

        # Detect faces using face_recognition for identification
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop over detected faces for identification
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, use the name of the first match
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                mark_attendance(name)  # Mark attendance if an employee is detected

            # Draw a bounding box with the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Employee Face Recognition", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
