# FitMaster – AI Coach

Project developed during the Seneca Hackathon 2025.  
FitMaster is an AI training coach that analyzes sports movements in real-time via webcam, corrects posture, and provides interactive feedback.  

Deployed on an Ubuntu VPS with SSH access → [Live demo](https://crotale.mirhosty.com/).

---

## Objective

Create an AI training coach capable of:  
- Detecting and analyzing sports movements (squat, push-up, burpee, etc.).  
- Correcting posture in real-time with visual and vocal feedback.  
- Providing a movement quality score.  
- Preventing injuries and motivating the user.  

---

## Tech Stack

- Backend → Flask (Python, port 8097).  
- Frontend → HTML / CSS / JavaScript.  
- AI Vision → [MediaPipe](https://developers.google.com/mediapipe) + [PyTorch](https://pytorch.org/).  
- Deployment → Ubuntu VPS + PM2 for Python process management.  

---

## Features

- Real-time detection of multiple exercises: squat, push-up, burpee (+ extensible).  
- Joint angle analysis to correct posture.  
- Visual feedback (skeleton overlay) + vocal feedback (“Straighten your back!”).  
- Assign a quality score for each movement.  

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

# Run Flask server on port 8097
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
AI & Detection → MediaPipe + PyTorch integration.

Frontend → HTML / CSS / JS.

Backend → Flask + API integration.

Pitch & Design → Storytelling and jury presentation.

Future Improvements
Add more exercises (lunges, jumping jack, plank, etc.).

Improve detection with filters and higher accuracy.

Implement a private robotic coach (kinesiologist robot).

Gamification (badges, challenges, progress tracking).
