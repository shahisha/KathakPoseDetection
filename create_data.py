import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Detect Ardhapataka hand gesture
def detect_ardhapataka(index_tip, index_base, middle_tip, middle_base, ring_tip, ring_base, little_tip, little_base, thumb_tip, thumb_base):
    # Check if all landmarks are detected
    if None in [index_tip, index_base, middle_tip, middle_base, ring_tip, ring_base, little_tip, little_base, thumb_tip, thumb_base]:
        return None

    angle_index = calculate_angle(index_tip, index_base, middle_tip)
    angle_middle = calculate_angle(middle_tip, middle_base, ring_tip)
    angle_ring = calculate_angle(ring_tip, ring_base, little_tip)
    angle_thumb_index = calculate_angle(thumb_tip, thumb_base, index_tip)
    angle_thumb_middle = calculate_angle(thumb_tip, thumb_base, middle_tip)
    angle_thumb_ring = calculate_angle(thumb_tip, thumb_base, ring_tip)
    angle_thumb_little = calculate_angle(thumb_tip, thumb_base, little_tip)
    print(angle_index, angle_middle, angle_ring, angle_thumb_little)

    return [angle_index, angle_middle, angle_ring, angle_thumb_index, angle_thumb_middle, angle_thumb_ring, angle_thumb_little]

# Path to the folder containing images
folder_path = r'C:\Users\ishas\Downloads\sign-language-detector-python-master\data\2'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# List to store angle values for each image
angle_values_list = []

# Iterate over all images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        
        # Load the image
        image = cv2.imread(image_path)

        # Process the image with MediaPipe Hands
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Initialize angle list for the current image
        angles = {}

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image.shape[0]))
                thumb_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image.shape[0]))
                index_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]))
                index_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image.shape[0]))
                middle_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image.shape[0]))
                middle_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image.shape[0]))
                ring_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image.shape[0]))
                ring_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image.shape[0]))
                little_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image.shape[0]))
                little_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image.shape[0]))
                
                # Calculate angles for Ardhapataka hand gesture
                angle_values = detect_ardhapataka(index_tip, index_base, middle_tip, middle_base, ring_tip, ring_base, little_tip, little_base, thumb_tip, thumb_base)
                
                if angle_values:
                    angles[filename] = angle_values

        angle_values_list.append(angles)

# Save angle values to a CSV file
print(angle_values_list)
csv_file = 'shikhara_final.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Angle Index', 'Angle Middle', 'Angle Ring', 'Angle Thumb Index', 'Angle Thumb Middle', 'Angle Thumb Ring', 'Angle Thumb Little'])
    
    for angles in angle_values_list:
        for filename, values in angles.items():
            writer.writerow([filename] + values)
