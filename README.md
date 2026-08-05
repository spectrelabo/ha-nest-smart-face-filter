# 🏠 ha-nest-smart-face-filter

> **[HA / CV] Event-driven face recognition for Google Nest & Home Assistant. Ultra-lightweight (<200MB RAM) using InsightFace ONNX.**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2024.x-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![ONNX Runtime](https://img.shields.io/badge/ONNXRuntime-CPU-orange)
[![PayPal](https://img.shields.io/badge/Donate-PayPal-00457C?style=flat-square&logo=paypal&logoColor=white)](https://paypal.me)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat-square&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/blueskies360)

---

## 💡 Why This Project Exists (Eliminating Notification Fatigue)

Smart cameras like **Google Nest** are great at detecting human motion, but they trigger a flood of repetitive notifications every single time a family member, neighbor, or frequent visitor walks by. 

**`ha-nest-smart-face-filter`** was born out of a real personal need: **to eliminate camera notification spam and receive alerts ONLY when an unknown or unexpected person is detected.**

By combining Home Assistant's local events with a lightweight local AI model (InsightFace ONNX), this project acts as an intelligent event filter—processing camera snapshots on-demand without expensive cloud subscriptions or continuous GPU streaming.

---

## 📌 Key Features

- **🎯 Smart Notification Filter**: Mutes repetitive alerts for recognized faces (family/friends) and notifies you **only** when an unknown person is detected.
- **⚡ Event-Driven & On-Demand**: Executes strictly when `binary_sensor.*_person_detected` changes to `on`, keeping idle CPU/RAM usage at 0.
- **🪶 Ultra-Lightweight (<200MB RAM)**: Uses InsightFace `buffalo_s` with `CPUExecutionProvider` on ONNX Runtime. Runs perfectly on Raspberry Pi or Mini PCs.
- **🚀 RAM Embedding Cache**: Fast L2 Cosine Similarity comparisons against local `.npy` face vectors.
- **🔔 Home Assistant Integration**: Dispatches native events (`unknown_person_detected` / `known_person_detected`) for flexible Home Assistant automations and push notifications.

---

## 🛠️ Project Structure

```plaintext
/config/face_recognition/
├── create_embeddings.py   # CLI tool to generate 512-d .npy face vectors
├── analyze.py             # Main event-driven analysis script
└── embeddings/            # Directory storing target .npy embedding files
/config/www/snapshots/
└── last_person.jpg        # Temporary snapshot saved by Home Assistant
```

---

## 🚀 Quick Start & Installation

### 1. Install Dependencies
Ensure Python 3 is available in your Home Assistant environment or container:
```bash
pip install -r requirements.txt
```

### 2. Generate Known Face Embeddings
Add a high-quality picture of a household member and extract their embedding vector:
```bash
python3 face_recognition/create_embeddings.py \
  --image /path/to/family_member.jpg \
  --name "mario_rossi" \
  --output-dir /config/face_recognition/embeddings
```

### 3. Add Home Assistant Configuration
Add the configuration snippet from [`ha_configuration_snippet.yaml`](./ha_configuration_snippet.yaml) to your `configuration.yaml` and `automations.yaml`.

---

## ☕ Support the Project

This is a 100% free and Open Source project born out of personal necessity to eliminate notification fatigue. If this project saved you time or reduced camera notification spam in your home, consider supporting its maintenance!

- 💳 **[Donate via PayPal](https://paypal.me)**
- ☕ **[Buy Me a Coffee](https://buymeacoffee.com)**

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
