Face Recognition Web Interface
This project is a web interface for real-time face recognition using Flask, face_recognition, and imageio. It allows users to upload a target image and a video, then processes the video to find and annotate faces that match the target image.

Features
Upload target image and video for face recognition.
Real-time face recognition with results displayed in the browser.
Annotated video output with recognized faces highlighted.
User-friendly web interface with responsive design.
Technologies Used
Flask: A lightweight WSGI web application framework in Python.
face_recognition: A library for face detection and recognition.
imageio: A library for reading and writing images and videos.
PIL (Pillow): Python Imaging Library for image processing.
HTML/CSS: For front-end design.
JavaScript: For real-time updates and client-side scripting.
Socket.IO: For real-time communication between the server and client.
Theory
Face Recognition Process
Face Detection: The first step involves detecting faces in an image or video frame. The face_recognition library uses Histogram of Oriented Gradients (HOG) and Convolutional Neural Networks (CNN) to detect faces.

Encoding Faces: Once faces are detected, the library encodes the faces into a 128-dimensional face embedding using a pre-trained neural network. This encoding is a numerical representation of the face's features.

Comparing Faces: The face encoding of the target image is compared with the encodings of the faces detected in the video frames. This comparison is done using a distance metric (Euclidean distance) to find matches.

Annotating Faces: When a face in the video matches the target image, the frame is annotated with rectangles around the detected faces and a label indicating a match.

Real-Time Processing
Web Interface: The Flask web application serves the HTML/CSS/JavaScript front-end, allowing users to upload images and videos.
Socket.IO: Enables real-time communication between the client and server, providing instant feedback on the face recognition process.
Video Processing: The video is processed frame by frame to detect and annotate faces. The processed frames are then compiled into an output video that is displayed on the web interface.
