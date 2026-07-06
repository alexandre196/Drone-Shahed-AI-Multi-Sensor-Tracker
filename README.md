# 🎯 Shahed Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-GPL--3.0-green)
![mAP@50 Shahed](https://img.shields.io/badge/mAP@50%20Shahed-99.5%25-brightgreen)
![mAP@50 Global](https://img.shields.io/badge/mAP@50%20Global-89.4%25-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-lightgrey)

**YOLOv8 real-time drone detection — Kalman tracking · PDF report · KML export**

*Système de détection temps réel de drones Shahed-136 par IA — développé par [alexandre196](https://github.com/alexandre196)*

</div>

---

## 🚀 Features

- ✅ **YOLOv8s fine-tuned** on Shahed dataset — mAP@50 = **99.5% on Shahed class** (89.4% global)
- ✅ **Multi-class classifier** : `bird` / `not` / `shahed` — optimized to minimize false positives
- ✅ **Multi-object Kalman tracking** — persistent drone ID, trajectory trail, velocity & direction
- ✅ **Behavioral analysis** — hovering, circling, fast approach, erratic motion detection
- ✅ **GPS-free geolocalization** — monocular distance + camera-heading-aware azimuth → lat/lon
- ✅ **Live radar mini-map** — real-time position display in GUI
- ✅ **3 input sources** — video file / webcam / RTSP IP camera (FLIR, Hikvision, Axis...)
- ✅ **Automated alerts** — audio alarm + email (Gmail SMTP) + push notification (Ntfy)
- ✅ **KML export** — Google Earth trajectory with geolocalized pins
- ✅ **CSV export** — for QGIS and GIS tools
- ✅ **Automated PDF report** — stats, charts, radar map, closest threat image
- ✅ **Temporal confirmation filter** — N consecutive frames before alarm (anti false-positive)
- ✅ **Configurable danger zone** — adjustable threshold in meters (default 300m)
- ✅ **Cross-platform** — Windows & macOS

---

## 📊 Model Performance (v2 — fine-tuned)

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| **shahed** | **99.5%** | 84.3% |
| bird | 83.9% | 52.1% |
| not | 86.3% | 57.9% |
| **ALL** | **89.4%** | 64.8% |

> Model v2 fine-tuned on 16,069 images including top-view, side-view and **bottom-view** Shahed-136 footage.  
> Training: 50 epochs · RTX 4070 Ti · 1h54

---

## 🖥️ Screenshot

<img width="3840" height="2086" alt="shahed_detector01" src="https://github.com/user-attachments/assets/0c4ffd19-4a07-4817-ae6a-10be6754fdce" />

---

## ⚙️ Installation

```bash
pip install ultralytics opencv-python numpy matplotlib reportlab Pillow
```

**Optional (audio alarm on Windows):**
```bash
winget install ffmpeg
```

**Optional (audio alarm on macOS):**
```bash
brew install ffmpeg
```

---

## 🚀 Usage

```bash
python drone_shahed_detector.py
```

1. Select your video source — **File / Webcam / RTSP stream**
2. Load the model (`runs/detect/shahed_detector/weights/best.pt`)
3. Set your camera GPS position, FOV, heading
4. Configure danger distance threshold (meters)
5. Click **LANCER LA DÉTECTION**

---

## 📁 Project Structure

```
Drone_Shaed_AI/
├── drone_shahed_detector.py   # Main detection system (GUI + pipeline)
├── entrainement_v2.py         # Fine-tuning script (transfer learning)
├── Annotate_dessous.py        # Manual annotation tool (YOLO format)
├── train_shahed.py            # Initial training script
├── download_shahed_dataset.py # Dataset download helper
├── chiffres.py                # Statistics utility
└── README.md
```

---

## 🧠 Architecture

```
Camera / Video / RTSP
        ↓
   YOLOv8s Inference (GPU/CPU)
        ↓
   Kalman Multi-Object Tracker
        ↓
   Behavioral Analysis
        ↓
   GPS-Free Geolocalization
        ↓
   Alert Pipeline (audio + email + push)
        ↓
   KML / CSV / PDF Export
```

---

## 🗂️ Model Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | `bird` | Birds — neutral, no alarm |
| 1 | `not` | Other objects — neutral |
| 2 | `shahed` | Shahed-136 kamikaze drone — **DANGER** |

---

## 📍 Geolocalization

The system estimates drone position **without GPS** using:
- Per-class real wingspan (Shahed-136 = **2.5m**)
- Camera FOV + auto-computed focal length
- Camera heading (0=North, 90=East, 180=South, 270=West)
- Pixel position → azimuth → lat/lon offset

Positions are exported as **KML** (Google Earth) and **CSV** (QGIS).

---

## 🔔 Alert Pipeline

| Trigger | Action |
|---------|--------|
| Shahed detected | Audio alarm |
| Distance < threshold | Email with drone image |
| Confirmed N frames | Ntfy push notification |
| End of analysis | Automated PDF report |

---

## 📄 License

GPL-3.0 — see [LICENSE](LICENSE)

---

## 👤 Author

**Alexandre Martin** — AM Consulting, France  
[GitHub](https://github.com/alexandre196) · [Website](https://www.amconsulting-formation.com)

> *"Every second of early warning saves lives."*
