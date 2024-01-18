# Evil System

This script uses OpenCV and MediaPipe for real-time human pose detection in a video feed captured from the device camera.  
The program tracks specific landmarks of the human body.  
The aim of the program is to find a human in the video feed.  
If human found, the program will display a message that the target has been detected and a countdown will start.  
The human needs to raise the hands above the head in 5 seconds.  
If that does not happen, the program will display a message that the target has been eliminated.  
If the user raises the hands above the head in 5 seconds, the target symbol is taken out and the countdown is cleared.

## How to set up:

Install the packages from the requirements.txt with the following command `pip3 install -r requirements.txt`

## How to run:

Run the program with the following command `python3 evil_system.py`  
The program will start capturing the video from the device camera.

## Demo:

![](evil-system-demo.gif)

## Authors:

Adam Łuszcz s22994  
Anna Rogala s21487

## Sources:

- https://developers.google.com/mediapipe/solutions
- https://github.com/googlesamples/mediapipe/blob/main/examples
