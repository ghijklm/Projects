import cv2
import mediapipe as mp
import math
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calculate_angle(a, b, c):
    # Calculate the angle between three points (a, b, c)
    angle_rad = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open a video capture object (you can replace 0 with the video file path if needed)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = pose.process(frame_rgb)

    # Extract keypoints
    keypoints = results.pose_landmarks.landmark

    # Calculate and print angles for upper arm, forearm, fingers, legs, head, and spine
    if keypoints:
        # Upper arm angles (example: left arm)
        left_shoulder = [keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value].x, keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        upper_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Forearm angles (example: left arm)
        forearm_angle = calculate_angle(left_elbow, left_wrist, [left_wrist[0] + 0.1, left_wrist[1]])

        # Finger angles (example: left hand)
        left_pinky = [keypoints[mp_pose.PoseLandmark.LEFT_PINKY.value].x, keypoints[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
        finger_angle = calculate_angle(left_wrist, left_pinky, [left_pinky[0] + 0.1, left_pinky[1]])

        # Leg angles (example: left leg)
        left_hip = [keypoints[mp_pose.PoseLandmark.LEFT_HIP.value].x, keypoints[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value].x, keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Head angle
        head_angle = calculate_angle([0, 0], [0, 1], [keypoints[mp_pose.PoseLandmark.NOSE.value].x, keypoints[mp_pose.PoseLandmark.NOSE.value].y])

        # Spine angle (angle between chest and stomach)
        chest = [keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        stomach = [keypoints[mp_pose.PoseLandmark.LEFT_HIP.value].x, keypoints[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        spine_angle = calculate_angle(chest, [chest[0], chest[1] + 0.1], stomach)

        # Print angles
        print(f"Upper Arm Angle: {upper_arm_angle:.2f} degrees")
        print(f"Forearm Angle: {forearm_angle:.2f} degrees")
        print(f"Finger Angle: {finger_angle:.2f} degrees")
        print(f"Leg Angle: {leg_angle:.2f} degrees")
        print(f"Head Angle: {head_angle:.2f} degrees")
        print(f"Spine Angle: {spine_angle:.2f} degrees")

    # Draw Mediapipe keypoints on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
