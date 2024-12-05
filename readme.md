# Eye Tracking and Attention Detection

This project uses **OpenCV** and **MediaPipe** to create an eye-tracking system capable of detecting if a person is looking away from the screen. It's useful for applications like exam monitoring or attention analysis.

---

## Features

- **Eye Aspect Ratio (EAR):** Measures eye openness for blinking or attention.
- **Looking Away Detection:** Identifies when the person is not looking at the screen using eye position and EAR thresholds.
- **Iris Position Tracking:** Visualizes iris positions with circles for intuitive feedback.
- **Frame Skipping:** Reduces processing overhead by analyzing every Nth frame.

---

## How It Works

1. **Camera Input:**  
   The application uses your webcam to capture live video frames.

2. **Face Mesh Detection:**  
   Using **MediaPipe**, the application detects 468 facial landmarks, including those around the eyes.

3. **EAR Calculation:**  
   - The **Eye Aspect Ratio (EAR)** is calculated for each eye based on specific eye landmarks.
   - EAR is determined by comparing the vertical and horizontal distances of the eye.
   - A low EAR value indicates that the eye is closed or blinking.

4. **Looking Away Detection Logic:**  
   - The EAR values are compared to a threshold to determine if the eyes are open or closed.
   - The position of the eyes relative to the screen's center is analyzed.
   - If the EAR is below the threshold or the eyes deviate significantly from the center, the user is considered to be "Looking Away."

5. **Iris Position Tracking:**  
   - The approximate position of the irises is calculated using eye landmarks.
   - These positions are displayed as yellow circles for intuitive feedback.

6. **Real-Time Feedback:**  
   - If the user is focused on the screen, the message "Looking At Screen" is displayed in green.
   - If the user looks away, the message "Looking Away" is displayed in red.

7. **Frame Skipping:**  
   - To reduce computational load, only every Nth frame is processed (configurable via `frame_skip_interval`).
