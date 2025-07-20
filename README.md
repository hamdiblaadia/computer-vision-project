# computer-vision-project
Computer vision model for Real_Time_ Liquid-Level-Detection on embedded systems
Computer Vision and Deep Learning Enabled Real-Time Liquid Level Detection and Measurement in Transparent Containers
# A. Problem Statement
Laboratories, particularly those in large oil companies, require a fast and reliable system for detecting and measuring liquid levels in containers. Current manual methods and some of the pure vision based methods are time-consuming, tedious, and prone to errors leading to delays in product development and increased risk of errors. There is a critical need for a fast and reliable system that can detect and measure liquid levels in real-time and improve the efficiency and reducing human errors in laboratory testing procedures.

Moreover, with the increase in the trend of Industry 4.0 the capability to detect and measure the liquid levels in the transparent flasks/test tubes is a crucial part in the whole perception system of an autonomous robot and by the use of the Computer vision and deeplearning we can easily help solve this problem. The final developed system can be used in various applications across various industries, including pharmaceutical, manufacturing and chemical processing making it a crucial component in the perception system of an autonomous robot in the Industry 4.0.
# B. Proposed Solution
To address the risks of  dosage and reduce human intervention, we proposed an intelligent, automated dosing system. This solution leverages embedded systems and computer vision to ensure real-time control, traceability, and precision.

The system is built around a hybrid architecture combining the ESP32 microcontroller for actuator control and a Raspberry Pi 5 for data processing, communication, and AI-based supervision.

Key components of the solution include:

🔹 Automation & Control
A set of motors and pumps are controlled via the ESP32 board using C++, ensuring precise liquid handling. Automation logic reduces manual tasks by over 80%.

🔹 Real-Time Computer Vision Monitoring
A computer vision model based on #DeepLearning using #PyTorch and #OpenCV is deployed on the Raspberry Pi. It continuously monitors the dosage process and detects any abnormal variations in liquid levels with an accuracy greater than 95%.

🔹 Remote Supervision & Communication
The system integrates #Firebase as a backend database and dashboard for logging all measurements and supervising operations remotely. A lightweight #MQTT protocol ensures real-time communication between the Raspberry Pi and ESP32 for synchronized control.

🔹 Modular & Scalable Architecture
All components are containerized using #Docker to ensure portability and ease of deployment.

This solution not only improves dosage safety but also allows medical personnel to track, monitor, and validate processes in real time.

# ⚙️ Methodology
# I. Dataset Creation on Roboflow
We captured multiple syringe images under different lighting and angle conditions.

Images were annotated and preprocessed using Roboflow

# II. Training YOLOv8 on Our Dataset
We trained a YOLOv8 model for real-time detection of syringe liquid levels.

The model was trained using PyTorch on a custom annotated dataset.
# 🚀 Results
Error Detection Accuracy: > 95%

Manual Intervention Reduction: ~80%

Real-Time Inference Speed: ~20 FPS on Raspberry Pi











