import cv2
import mediapipe as mp
import math

# --- setup mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- calculate angle between 3 points ---
def get_angle(a, b, c):
    ab = [a.x - b.x, a.y - b.y]
    cb = [c.x - b.x, c.y - b.y]
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    norm_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    norm_cb = math.sqrt(cb[0]**2 + cb[1]**2)
    angle = math.acos(dot / (norm_ab * norm_cb))
    return math.degrees(angle)

# --- detect which fingers are open ---
def get_finger_states(landmarks, hand_label):
    fingers = []

    # thumb (x-axis comparison)
    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    # other fingers (use angle of joints)
    angles = {
        "index": get_angle(landmarks[5], landmarks[6], landmarks[8]),
        "middle": get_angle(landmarks[9], landmarks[10], landmarks[12]),
        "ring": get_angle(landmarks[13], landmarks[14], landmarks[16]),
        "pinky": get_angle(landmarks[17], landmarks[18], landmarks[20]),
    }

    for name in ["index", "middle", "ring", "pinky"]:
        angle = angles[name]
        fingers.append(1 if angle > 160 else 0)  # open if nearly straight

    return fingers

# --- detect gesture based on finger state ---
def gesture_name(fingers):
    thumb, index, middle, ring, pinky = fingers

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Hand"
    elif thumb == 1 and index == middle == ring == pinky == 0:
        return "Thumbs Up"
    elif middle == 1 and thumb == index == ring == pinky == 0:
        return "Middle Finger"
    elif thumb == 1 and pinky == 1 and index == middle == ring == 0:
        return "Rock Sign"
    elif thumb == 1 and index == 1 and pinky == 1 and middle == ring == 0:
        return "I Love You"
    elif index == 1 and middle == 1 and pinky == 1 and thumb == ring == 0:
        return "Vulcan Salute"
    elif index == 1 and middle == ring == pinky == 0:
        return "1"
    elif index == 1 and middle == 1 and ring == pinky == 0:
        return "2"
    elif index == 1 and middle == 1 and ring == 1 and pinky == 0:
        return "3"
    elif index == 1 and middle == 1 and ring == 1 and pinky == 1:
        return "4"
    else:
        return "Unknown"

# --- run camera ---
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_label = results.multi_handedness[idx].classification[0].label  # Right or Left
            fingers = get_finger_states(hand_landmarks.landmark, hand_label)
            gesture = gesture_name(fingers)

            cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
