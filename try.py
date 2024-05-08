import cv2
import mediapipe as mp
import numpy as np

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
    angle_index = calculate_angle(index_tip, index_base, middle_tip)
    angle_middle = calculate_angle(middle_tip, middle_base, ring_tip)
    angle_ring = calculate_angle(ring_tip, ring_base, little_tip)

    if angle_index < 90 and angle_middle > 90 and angle_ring > 90:
        return True
    else:
        return False

# Detect Pataka hand gesture
def detect_pataka(index_tip, index_base, middle_tip, middle_base, ring_tip, ring_base, little_tip, little_base, thumb_tip, thumb_base):
    angle_index = calculate_angle(index_tip, index_base, middle_tip)
    angle_middle = calculate_angle(middle_tip, middle_base, ring_tip)
    angle_ring = calculate_angle(ring_tip, ring_base, little_tip)
    angle_thumb_index = calculate_angle(thumb_tip, thumb_base, index_tip)
    angle_thumb_middle = calculate_angle(thumb_tip, thumb_base, middle_tip)
    angle_thumb_ring = calculate_angle(thumb_tip, thumb_base, ring_tip)
    angle_thumb_little = calculate_angle(thumb_tip, thumb_base, little_tip)

    if angle_thumb_index < 20 and angle_index < 20 and angle_middle < 20 and angle_ring < 20:
        thumb_distance_index = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
        thumb_distance_middle = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))
        thumb_distance_ring = np.linalg.norm(np.array(thumb_tip) - np.array(ring_tip))
        thumb_distance_little = np.linalg.norm(np.array(thumb_tip) - np.array(little_tip))
        
        if thumb_distance_index > 50 and thumb_distance_middle > 50 and thumb_distance_ring > 50 and thumb_distance_little > 50:
            return True
    return False

# Detect Shikhara hand gesture
def detect_shikhara(index_tip, index_base, middle_tip, middle_base, ring_tip, ring_base, little_tip, little_base, thumb_tip, thumb_base):
    angle_thumb_index = calculate_angle(thumb_tip, thumb_base, index_tip)
    angle_thumb_middle = calculate_angle(thumb_tip, thumb_base, middle_tip)
    angle_thumb_ring = calculate_angle(thumb_tip, thumb_base, ring_tip)
    angle_thumb_little = calculate_angle(thumb_tip, thumb_base, little_tip)

    if angle_thumb_index > 120 and angle_thumb_middle > 120 and angle_thumb_ring > 120 and angle_thumb_little > 120:
        return True
    else:
        return False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]))
            thumb_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * frame.shape[0]))
            index_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))
            index_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * frame.shape[0]))
            middle_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0]))
            middle_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame.shape[0]))
            ring_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * frame.shape[0]))
            ring_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * frame.shape[0]))
            little_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * frame.shape[0]))
            little_base = (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * frame.shape[0]))
            
            # Detect Ardhapataka hand gesture
            ardhapataka_detected = detect_ardhapataka(index_tip, index_base, middle_tip, middle_base, ring_tip, ring_base, little_tip, little_base, thumb_tip, thumb_base)
            if ardhapataka_detected:
                cv2.putText(frame, "Ardhapataka Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Detect Pataka hand gesture
            pataka_detected = detect_pataka(index_tip, index_base, middle_tip, middle_base, ring_tip, ring_base, little_tip, little_base, thumb_tip, thumb_base)
            if pataka_detected:
                cv2.putText(frame, "Pataka Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Detect Shikhara hand gesture
            shikhara_detected = detect_shikhara(index_tip, index_base, middle_tip, middle_base, ring_tip, ring_base, little_tip, little_base, thumb_tip, thumb_base)
            if shikhara_detected:
                cv2.putText(frame, "Shikhara Detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
