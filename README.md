# FitMaster – AI Training Coach with Chatbot

<img src="https://crotale.mirhosty.com/static/Fitmaster-logo.png" alt="Logo du projet" width="200"/>

Project developed during the Seneca Hackathon 2025.  
FitMaster is an AI-powered training coach that analyzes sports movements in real-time via webcam. It provides instant visual and vocal feedback, evaluates movement quality, helps prevent injuries, and includes an **integrated chatbot** that answers users' questions about exercises, posture, and general workout guidance.  

Deployed on an Ubuntu VPS with SSH access (Increase performances if we scale our VPS) → [Live demo](https://crotale.mirhosty.com/).

---

## Images Capture (Pc)
<img src="https://crotale.mirhosty.com/static/image5.jpg" alt="Capture PC" width="500"/>

## Images Capture (Smartphone)
<p align="center">
  <img src="https://crotale.mirhosty.com/static/image2.jpg" alt="Capture Smartphone" width="200"/>
  <img src="https://crotale.mirhosty.com/static/image3.jpg" alt="Capture Smartphone" width="200"/>
  <img src="https://crotale.mirhosty.com/static/image6.jpg" alt="Capture Smartphone" width="200"/>
</p>



## Objective

Create an AI training coach capable of:  
- Detecting and analyzing sports movements (squat, push-up, burpee, etc.).  
- Correcting posture in real-time with visual and vocal feedback.  
- Providing a movement quality score.  
- Answering user questions through an integrated chatbot about exercises, fitness tips, and training routines.  
- Preventing injuries and motivating the user.  

---

## Tech Stack


- **Backend** : Python **Flask** + **Flask‑Socket.IO** (transport temps réel), évent. **Eventlet** comme serveur async.
- **Frontend** : HTML/CSS/JS, **Socket.IO client**, Canvas/WebGL pour affichage, WebAudio pour TTS (optionnel).
- **Vision** : **MediaPipe Pose** (extraction points clés), **OpenCV** (traitement), **NumPy** (angles/metrics).
- **(Optionnel)** Modèles **TFLite**/accélération CPU (XNNPACK) si vous utilisez un modèle custom.
- **I/O** : Webcam côté navigateur (getUserMedia) ou flux image envoyé au backend (selon variante).
- **Fonctions** : Détection de posture, calcul d’angles (genoux, hanches, coudes, épaules…), logique de répétitions, **score qualité de mouvement** basé sur la cinématique, retour vocal/texte.

---

## 📦 Dépendances et versions testées (pinnées)

Créez un **venv Python 3.10–3.11** (évitez 3.12 si vous utilisez Mediapipe < 0.10.14).

## requirements.txt (Compatibles versions)

```txt
Flask==3.0.3
python-socketio==5.11.3
Flask-SocketIO==5.4.1
eventlet==0.36.1
mediapipe==0.10.14
opencv-python==4.10.0.84
numpy==2.0.2
scipy==1.13.1
sounddevice==0.4.7
pyttsx3==2.98
tflite-runtime==2.14.0
```

## Features

- Real-time detection of multiple exercises: squat, push-up, burpee (+ extensible)  
- Joint angle analysis to correct posture  
- Visual feedback (skeleton overlay) + vocal feedback (“Straighten your back!”)  
- Movement quality score for each exercise  
- **Integrated chatbot** to answer questions about exercises, posture, and fitness tips  

---

## Local Installation

### Prerequisites
- [Python 3.10+](https://www.python.org/downloads/)  
- [pip](https://pip.pypa.io/en/stable/) or conda  
- Modern web browser (Chrome, Edge, Firefox)  

### Steps
```bash
# Clone the repository
git clone https://github.com/chaton2325/Seneca-Hack.git
cd Seneca-Hack

# Navigate to backend directory
cd version2/crotale

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask server on port 8097 (Or choose your port)
flask run --host=0.0.0.0 --port=8097
Frontend → open index.html in your browser or run with your preferred server.

VPS Deployment
SSH Connection
bash
Copier le code
ssh crotale@147.93.116.24
(or use your VPS IP and username)

Navigate to project root
bash
Copier le code
cd /path/to/version2/crotale
Start Flask backend in the background using PM2
If PM2 is not installed:

bash
Copier le code
sudo npm install -g pm2
bash
Copier le code
pm2 start crotale.py --interpreter python3
Access the application
Use the VPS IP directly or configure a domain name.

Enter the password if prompted.

Live demo: https://crotale.mirhosty.com/

Team
AI & Detection → MediaPipe + PyTorch integration

Frontend → HTML / CSS / JS

Backend → Flask + API + chatbot integration

Pitch & Design → Storytelling and jury presentation

Future Improvements
Add more exercises (lunges, jumping jack, plank, etc.)

Improve detection with filters and higher accuracy

Enhance chatbot intelligence for personalized workout guidance

Implement a private robotic coach (kinesiologist robot)

Gamification (badges, challenges, progress tracking)
