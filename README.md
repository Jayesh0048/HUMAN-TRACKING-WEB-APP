🧍‍♂️ Human Tracking & Alert Notification Web Application

A real-time human tracking web application that identifies a specific person using a reference image and monitors nearby camera feeds to detect their presence.
When the person is detected, the system automatically sends an email alert to the user or administrator.

🚀 Project Overview

This application is designed to track a particular individual by comparing a provided reference image with live or nearby camera feeds.
Once the system recognizes the person in the camera frame, it triggers an instant email notification, making it useful for security, monitoring, and alert-based applications.

🎯 Key Features

Upload reference image of a person
Real-time human detection using camera feed
Face matching with stored reference image
Automatic email alerts on successful detection
Web-based interface for monitoring and control

📸 How the System Works
1️⃣ Reference Image Registration

User uploads an image of the person to be tracked
Facial features are extracted and stored
The image is linked to a specific tracking request

2️⃣ Live Camera Monitoring

The system continuously monitors nearby or connected cameras
Each frame is analyzed for human presence
Detected faces are compared with the reference image

3️⃣ Person Detection & Matching

If a match is found:
Identity is confirmed
Timestamp and camera details are recorded
Detection confidence is evaluated

4️⃣ Email Alert System

Once detection is confirmed:
Email notification is sent to the user/admin
Alert includes detection time and camera information
Prevents repeated alerts for the same event

📧 Notification System

Email-based alert mechanism
Configurable recipient (user or admin)
Triggered only on valid detections
Designed to avoid duplicate alerts

🖥️ Web Dashboard

The web interface allows users to:
Upload and manage reference images
Start or stop tracking sessions
View detection status
Monitor camera feed
Manage email alert settings

🏗️ Tech Stack (High-Level)
Backend: FastAPI / Django
Frontend: Web-based UI
Computer Vision: Face detection and recognition
AI/ML: Face embedding and similarity matching
Email Service: SMTP / Email API

🔐 Security & Privacy

Secure handling of uploaded images
Role-based access to tracking controls
No unnecessary storage of camera footage
Privacy-aware design

📌 Use Cases

Security and surveillance systems
Missing person tracking
Office or campus monitoring
Restricted area access alerts
Smart surveillance applications

🌱 Future Enhancements

Multi-camera tracking
Real-time SMS / push notifications
Cloud deployment
Face re-identification across cameras
Improved accuracy with deep learning models

👨‍💻 Author

Jayesh Naidu
Machine Learning Engineer | Computer Vision Enthusiast
Focused on real-time AI systems and intelligent surveillance solutions

⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub.
