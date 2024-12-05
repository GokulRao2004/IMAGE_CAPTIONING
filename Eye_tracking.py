import cv2
import mediapipe as mp
import math

# Initialize camera and MediaPipe Face Mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Define a function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    if len(eye) < 4:  # Ensure we have enough landmarks for EAR calculation
        return 0.0
    A = math.dist(eye[1], eye[3])  # Vertical distance (top to bottom)
    C = math.dist(eye[0], eye[2])  # Horizontal distance (inner to outer)
    ear = A / C if C != 0 else 0.0
    return ear

# Define a function to detect if the person is looking away
def is_looking_away(landmarks, frame_w, frame_h):
    if len(landmarks) < 468:  # Check for the number of landmarks
        return False  # Not enough landmarks detected

    try:
        # Left eye landmarks
        left_eye = [
            (landmarks[474].x * frame_w, landmarks[474].y * frame_h),  # top
            (landmarks[475].x * frame_w, landmarks[475].y * frame_h),  # bottom
            (landmarks[476].x * frame_w, landmarks[476].y * frame_h),  # outer corner
            (landmarks[477].x * frame_w, landmarks[477].y * frame_h)   # inner corner
        ]

        # Right eye landmarks using landmark 469, 470, 471, and 472 for more accuracy
        right_eye = [
            (landmarks[469].x * frame_w, landmarks[469].y * frame_h),  # top
            (landmarks[470].x * frame_w, landmarks[470].y * frame_h),  # bottom
            (landmarks[471].x * frame_w, landmarks[471].y * frame_h),  # outer corner
            (landmarks[472].x * frame_w, landmarks[472].y * frame_h)   # inner corner
        ]

        # Calculate EAR for both eyes
        ear_left = eye_aspect_ratio(left_eye) if len(left_eye) == 4 else 0.0
        ear_right = eye_aspect_ratio(right_eye) if len(right_eye) == 4 else 0.0

        # Increased sensitivity thresholds
        ear_threshold = 0.1  # Lowered threshold for increased sensitivity
        deviation_threshold = 20  # Lowered threshold for increased sensitivity

        # Calculate the center of the face for reference
        center_x = frame_w // 2

        # Calculate distances from the eye landmarks to the center
        left_distance = abs(left_eye[3][0] - center_x)  # x-coordinate of the inner left eye corner
        right_distance = abs(right_eye[2][0] - center_x)  # x-coordinate of the outer right eye corner

        # Check conditions for looking away
        if (ear_left < ear_threshold or ear_right < ear_threshold) or (left_distance > deviation_threshold and right_distance > deviation_threshold):
            return True
    except Exception as e:
        print("Error in eye landmark processing:", e)

    return False

def draw_iris(landmarks, frame, frame_w, frame_h):
    # Approximate positions for the irises
    iris_left = (landmarks[474].x * frame_w, landmarks[474].y * frame_h)  # Approximate position for left iris
    iris_right = (landmarks[469].x * frame_w, landmarks[469].y * frame_h)  # Approximate position for right iris

    # Draw iris positions
    cv2.circle(frame, (int(iris_left[0]), int(iris_left[1])), 5, (0, 255, 255), -1)  # Draw left iris
    cv2.circle(frame, (int(iris_right[0]), int(iris_right[1])), 5, (0, 255, 255), -1)  # Draw right iris

# Frame processing interval (process every Nth frame)
frame_skip_interval = 5  # Adjust to 2 or 3 depending on how often you want to process frames
frame_counter = 0

while True:
    _, frame = cam.read()
    if frame is None:
        print("Failed to capture frame")
        break

    frame_counter += 1

    # Skip frames to reduce processing
    if frame_counter % frame_skip_interval != 0:
        continue

    frame = cv2.flip(frame, 1)

    # Convert to grayscale for analysis (single channel)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Draw the eye landmarks if available
        for i in range(474, 478):  # Left eye landmarks
            if i < len(landmarks):
                x = int(landmarks[i].x * frame_w)
                y = int(landmarks[i].y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))

        # Draw the right eye landmarks (469, 470, 471, 472)
        for i in range(469, 473):  # Correcting the range for right eye landmarks
            if i < len(landmarks):
                x = int(landmarks[i].x * frame_w)
                y = int(landmarks[i].y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255))  # Draw right eye landmarks

        # Draw irises
        draw_iris(landmarks, frame, frame_w, frame_h)

        # Check if the person is looking away
        if is_looking_away(landmarks, frame_w, frame_h):
            cv2.putText(frame, "Looking Away", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Looking At Screen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
