# VisualVroom - ViT-based Deaf Driver Assistance Wearable App

[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red)](https://youtu.be/BQmdU7jwddo)
[![Working Prototype](https://img.shields.io/badge/Prototype-YouTube-red)](https://youtube.com/shorts/Jkzg4zz500I?feature=share)
[![Frontend Repo](https://img.shields.io/badge/Frontend-GitHub-blue)](https://github.com/GDG-SKKU/VisualVroom_Android_GDG)
[![Backend Repo](https://img.shields.io/badge/Backend-GitHub-blue)](https://github.com/GDG-SKKU/VisualVroom_Backend_GDG)

## 🚗 Overview

**"Hear the road, see the road."** VisualVroom is an innovative wearable application that pairs smartphones and smartwatches to provide deaf drivers with real-time visual and haptic alerts for traffic sounds. Using AI-powered audio analysis, the app detects emergency vehicles, motorcycles, and car horns while determining their direction, delivering critical safety information through visual cues and vibration patterns.

## ✨ Key Features

### 🔊 Recognition and Identification of Traffic Sounds
- **Vehicle Type Detection**: Distinguishes between sirens, motorcycles, and car horns
- **Directional Awareness**: Uses smartphone stereo microphones to identify sound direction (left/right)
- **Real-time Processing**: Continuous audio monitoring with instant alerts

### 🤟 Speech-to-Sign Language Conversion
- **Live Transcription**: Converts speech to text using Google Speech-to-Text
- **Sign Language Generation**: Creates sign language images using Google Gemini
- **Accessibility Support**: Helps deaf drivers communicate with law enforcement and others

## 🏗️ Architecture

![image](https://github.com/user-attachments/assets/466a48a6-b27a-46c1-9732-dedcbf8436ff)


### Audio Processing Pipeline

1. **Audio Capture**: Stereo microphones capture ambient sound
2. **Feature Extraction**: 
   - Generate spectrograms for frequency analysis
   - Extract MFCC (Mel-Frequency Cepstral Coefficients) features
   - Stitch features into a unified image representation
3. **AI Classification**: Vision Transformer (ViT) model processes the audio-visual representation
4. **Direction Detection**: Amplitude analysis determines left/right orientation
5. **Alert Delivery**: Results sent to smartwatch for haptic and visual feedback

## 🛠️ Technology Stack

### Frontend (Android)
- **Language**: Java
- **IDE**: Android Studio
- **Framework**: Android SDK (API Level 30+)
- **Wearable**: WearOS by Google
- **UI Components**: 
  - Lottie animations
  - Material Design components
  - ViewPager2 for tabbed interface

### Backend
- **Framework**: FastAPI (Python)
- **AI/ML**: 
  - PyTorch with Vision Transformer (ViT)
  - librosa for audio processing
  - Whisper AI for speech-to-text
- **Audio Processing**: 
  - soundfile, pydub for audio manipulation
  - numpy for numerical operations
- **Infrastructure**: Google Compute Engine

### APIs & Services
- **Google Speech-to-Text**: Audio transcription
- **Google Gemini**: Sign language image generation
- **Google Wearable API**: Watch communication

## 📱 Application Structure

### Mobile App (`mobile/`)
```
mobile/
├── src/main/java/edu/skku/cs/visualvroomandroid/
│   ├── MainActivity.java                 # Main activity with tab navigation
│   ├── AudioRecorderFragment.java        # Sound detection interface
│   ├── SpeechToTextFragment.java         # Speech-to-sign conversion
│   ├── AudioRecorder.java               # Audio recording logic
│   ├── AudioRecordingService.java       # Background audio service
│   ├── WearNotificationService.java     # Watch communication
│   └── dto/                             # Data transfer objects
├── src/main/res/
│   ├── layout/                          # UI layouts (portrait/landscape)
│   ├── raw/                             # Lottie animation files
│   └── values/                          # App resources
└── AndroidManifest.xml                  # App permissions and services
```

### Wear App (`wear/`)
```
wear/
├── src/main/java/edu/skku/cs/visualvroomandroid/presentation/
│   └── MainActivity.java                # Watch app main activity
├── src/main/res/layout/
│   └── activity_main.xml               # Watch UI layout
└── AndroidManifest.xml                 # Watch app manifest
```

### Backend (`direction/backend/`)
```
backend/
└── main.py                             # FastAPI server with:
                                        # - ViT model inference
                                        # - Audio processing pipeline
                                        # - Whisper transcription
                                        # - API endpoints
```

## 🚀 Getting Started

### Prerequisites
- Android Studio (latest version)
- Android device with API level 30+
- WearOS smartwatch (optional but recommended)
- Python 3.8+ (for backend development)

### Installation

#### Frontend Setup
1. Clone the frontend repository:
```bash
git clone https://github.com/GDG-SKKU/VisualVroom_Android_GDG.git
cd VisualVroom_Android_GDG
```

2. Open in Android Studio and build the project

3. Grant required permissions:
   - Microphone access
   - Location access
   - Notification permissions

#### Backend Setup
1. Clone the backend repository:
```bash
git clone https://github.com/GDG-SKKU/VisualVroom_Backend_GDG.git
cd VisualVroom_Backend_GDG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
python main.py
```

### Usage

1. **Sound Detection Mode**:
   - Launch the app and navigate to "Audio Recorder" tab
   - Tap the microphone button to start continuous monitoring
   - Visual alerts appear on phone, haptic feedback on watch

2. **Speech-to-Sign Mode**:
   - Navigate to "Speech to Text" tab
   - Tap record button and speak
   - View transcribed text and generated sign language images

## 🔧 Configuration

### Audio Settings
- **Sample Rate**: 16kHz for speech, 48kHz for sound detection
- **Channels**: Stereo recording for directional detection
- **Processing Interval**: 3-second windows for continuous monitoring
- **Confidence Threshold**: 97% for high-accuracy alerts

### Model Parameters
- **Architecture**: Vision Transformer (ViT-B/16)
- **Classes**: 6 total (Siren_L, Siren_R, Bike_L, Bike_R, Horn_L, Horn_R)
- **Input Size**: 224x224 grayscale images
- **Checkpoint**: `feb_25_checkpoint.pth`

## 💰 Cost Estimation

### Google Speech-to-Text
- **Free Tier**: 0-60 minutes per month
- **Paid Tier**: $0.016/minute beyond 60 minutes
- **Monthly Estimate**: ~$4 per driver (based on 300 minutes usage)

### Google Gemini
- **Cost**: Free for sign language generation (as assumed)

