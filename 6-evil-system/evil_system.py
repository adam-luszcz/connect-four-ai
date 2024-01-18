"""
Evil System
---
This script uses OpenCV and MediaPipe for real-time human pose detection in a video feed.
The program tracks specific landmarks of the human body.
The aim of the program is to find a human in the video feed.
If human found, the program will display a message that the target has been detected and a countdown will start.
The human needs to raise the hands above the head in 5 seconds.
If that does not happen, the program will display a message that the target has been eliminated.
If the user raises the hands above the head in 5 seconds, the target symbol is taken out and the countdown is cleared.


How to set up:
---
Run `pip3 install -r requiremeners.txt` to install the required libraries.

How to run:
---
Run `python3 evil_system.py` to run the script.

Authors: Adam Łuszcz, Anna Rogala
"""
import cv2
import mediapipe as mp
import math
import time


class PoseDetector:
    """
    Class for detecting human poses in a video feed using MediaPipe.

    Attributes:
        pose (mediapipe solution): The MediaPipe pose solution.
        time_det (bool): Flag to indicate if the time detection is active.
        start_time (float): Start time of the detection.
    """

    def __init__(self):
        """
        Initializes the PoseDetector class with MediaPipe pose solution.
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.time_det = False
        self.start_time = 0

    def draw_target(self, image, center, size, color=(0, 0, 255)):
        """
        Draws a target symbol on the image at a specified center and size.

        Args:
            image (ndarray): The image where the target will be drawn.
            center (tuple): The center coordinates of the target.
            size (int): The size of the target.
            color (tuple, optional): The color of the target. Defaults to red.
        """
        cv2.circle(image, center, size, color, 4)
        cv2.line(image, (center[0] - size, center[1]), (center[0] + size, center[1]), color, 4)
        cv2.line(image, (center[0], center[1] - size), (center[0], center[1] + size), color, 4)

    def put_text_center(self, image, text, y_pos, font_scale, color, thickness):
        """
        Places a text at the center of the image.

        Args:
            image (ndarray): The image on which text will be placed.
            text (str): The text to be placed.
            y_pos (int): The y-coordinate for the text position.
            font_scale (float): The scale of the font.
            color (tuple): The color of the text.
            thickness (int): The thickness of the text.
        """
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x_pos = int((image.shape[1] - text_size[0]) / 2)
        cv2.putText(image, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def process_pose(self, image):
        """
        Processes the pose in the given image.

        Args:
            image (ndarray): The image to be processed for pose detection.

        Returns:
            ndarray: The processed image.
        """
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            center = (int(nose.x * image.shape[1]), int(nose.y * image.shape[0]))
            face_size = int(
                abs(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x - landmarks[
                    self.mp_pose.PoseLandmark.RIGHT_EAR].x) *
                image.shape[1])

            color = (255, 255, 255)
            if left_wrist.y < nose.y and right_wrist.y < nose.y:
                self.time_det = False
            else:
                if not self.time_det:
                    self.start_time = time.time()
                    self.time_det = True
                elif time.time() - self.start_time >= 5:
                    self.put_text_center(image, 'Target eliminated!', 50, 1, (255, 0, 0), 2)
                    color = (255, 0, 0)
                else:
                    elapsed_time = str(math.ceil(5 - (time.time() - self.start_time)))
                    self.put_text_center(image, 'Target detected!', 50, 1, (255, 255, 255), 2)
                    self.put_text_center(image, f'Shooting in: {elapsed_time}', 80, 1, (255, 255, 255), 2)
                self.draw_target(image, center, face_size, color)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        return image


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while cap.isOpened():
        _, image = cap.read()
        image = cv2.flip(image, 1)

        processed_image = detector.process_pose(image)

        cv2.imshow('Evil Game', processed_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
