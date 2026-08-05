# ha-nest-smart-face-filter

> Kill the Notification Fatigue. Event-driven, ultra-lightweight (<200MB RAM) face recognition for Google Nest & Home Assistant. Stop getting alerted every time your neighbour walks by.

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/default)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PayPal](https://img.shields.io/badge/Donate-PayPal-00457C?style=flat&logo=paypal&logoColor=white)](https://paypal.me/alessiopicco)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/blueskies360)

---

## The Problem & The Fix

Got a Google Nest Doorbell or Camera pointed at a hallway, porch, or shared entrance? 
Every time a neighbor, the postman, or your partner walks past, your phone buzzes. All. Day. Long.

You don't want to turn off notifications—you actually want to know if a stranger is lurking at your door. But Google's native AI won't let you run local, custom facial filtering on Home Assistant without paying through the nose for cloud resources or melting a Raspberry Pi.

**The Fix:** 
A zero-cloud, local-first, event-driven facial filter using **InsightFace (buffalo_s)** on **ONNX Runtime (CPU)**.
- **Zero Idle Overhead:** It sleeps completely until Google Nest fires a `camera_person` event.
- **Lightning Fast:** Processes an HD frame in ~350ms on standard X86/ARM CPUs.
- **Microfootprint:** Consumes < 200MB RAM during inference.

---

## Architecture & Workflow

```
+-----------------+       1. Person Event       +----------------------+
|  Google Nest    | --------------------------> |    Home Assistant    |
|  Doorbell       |                             | (Event Automation)   |
+-----------------+                             +----------+-----------+
                                                           | 2. Trigger Snapshot & Python
                                                           v
+-----------------+       4. Match?             +----------------------+
| Actionable      | <--- YES: Silence / Log     |  python3             |
| Notification    |      NO:  Notify + Action   |  analyze.py          |
| & HA Dashboard  |                             | (InsightFace ONNX)   |
+-----------------+                             +----------------------+
```

---

## Complete End-to-End Setup Guide

### Step 1: Google Cloud Console Configuration

1. Go to [Google Cloud Console](https://console.cloud.google.com/) and create a project (e.g., `your-gcp-project-id`).
2. Go to **APIs & Services -> Library** and enable the following APIs:
   - **Smart Device Management API**
   - **Cloud Pub/Sub API**
3. Go to **APIs & Services -> Credentials** and click **Create Credentials -> OAuth client ID**:
   - Application type: **Web Application**
   - Name: `Home Assistant Nest`
   - Add the following **Authorized redirect URIs**:
     - `https://my.home-assistant.io/redirect/oauth`
     - `https://<YOUR_HA_DOMAIN>/auth/external/callback` (replace `<YOUR_HA_DOMAIN>` with your Nabu Casa or duckdns domain if applicable)
4. Save your **Client ID** and **Client Secret**.

---

### Step 2: Google Device Access Console & Pub/Sub Setup

1. Register at [Google Device Access Console](https://console.nest.google.com/device-access/) ($5 one-time fee).
2. Create a new Device Access Project.
3. Paste the **OAuth Client ID** generated in Step 1.
4. Note your **Device Access Project ID** (looks like `209c50d2-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
5. Open [Google Cloud Console - Pub/Sub Topics](https://console.cloud.google.com/cloudpubsub/topic/list).
6. Click **Create Topic** and name it `nest-events`.
   - Full topic string: `projects/your-gcp-project-id/topics/nest-events`
7. Click on the `nest-events` topic, open the **Permissions** sidepanel, and click **Add Principal**:
   - New Principal: `dev-access-api@system.gserviceaccount.com` (or `sdm-publisher@googlegroups.com`)
   - Role: **Pub/Sub -> Pub/Sub Publisher**
8. Go back to Device Access Console and paste `projects/your-gcp-project-id/topics/nest-events` into the Pub/Sub topic field.

---

### Step 3: Connect Nest to Home Assistant

1. In Home Assistant, navigate to **Settings -> Devices & Services -> Add Integration**.
2. Select **Nest**.
3. When prompted, enter:
   - **GCP Project ID:** `your-gcp-project-id`
   - **Device Access Project ID:** `209c50d2-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
   - **OAuth Client ID & Secret** (if prompted).
4. Complete the Google OAuth login workflow in your browser and authorize access to your Nest devices.
5. Verify that your camera entity is listed under Home Assistant (e.g., `camera.front_door`) along with the event entity `event.front_door_motion`.

---

### Step 4: Local Folder & Python Environment Setup

SSH into your Home Assistant host (or use Terminal / SSH Addon):

1. Create the project directories:
```bash
mkdir -p /config/face_recognition/embeddings
mkdir -p /config/www
```

2. Install Python dependencies:
```bash
pip3 install --no-cache-dir \
  insightface \
  onnxruntime \
  opencv-python-headless \
  numpy
```

---

### Step 5: Generate Known Face Embeddings

Convert face photos of household members into 512-dimensional vector files (`.npy`):

1. Place a clear, front-facing photo of a known person (e.g. `john.jpg`) into `/config/www/`.
2. Run the embedding extraction script:
```bash
python3 /config/face_recognition/create_embeddings.py \
  --image /config/www/john.jpg \
  --name "John"
```
3. The script creates `/config/face_recognition/embeddings/John.npy`. The raw image can now be deleted.

---

### Step 6: Home Assistant Configuration & Automations

1. Add the following to your `configuration.yaml`:

```yaml
homeassistant:
  allowlist_external_dirs:
    - "/config/www"

shell_command:
  process_face_recognition: "python3 /config/face_recognition/analyze.py"
  learn_latest_face: "python3 /config/face_recognition/learn_face.py --name '{{ name }}'"
```

2. Add the main automation to `automations.yaml`:

```yaml
- id: nest_camera_face_recognition_pipeline
  alias: "Nest Camera - Face Recognition Pipeline"
  description: "Triggers on Nest person event, captures snapshot, and executes face recognition."
  mode: single
  max_exceeded: silent
  trigger:
    - platform: state
      entity_id: event.front_door_motion
      attribute: event_type
      to: "camera_person"
  action:
    - service: camera.snapshot
      target:
        entity_id: camera.front_door
      data:
        filename: "/config/www/last_person.jpg"
    - service: shell_command.process_face_recognition
```

3. Add the notification automation for unknown faces to `automations.yaml`:

```yaml
- id: notify_on_unknown_person_detected
  alias: "Nest Camera - Unknown Person Notification"
  description: "Sends an actionable notification when an unknown person is detected."
  trigger:
    - platform: event
      event_type: unknown_person_detected
  action:
    - service: notify.notify
      data:
        title: "Unknown Person Detected!"
        message: "Unrecognized face detected at the front door."
        data:
          url: "/face-recognition/home"
          image: "/local/last_person.jpg"
          actions:
            - action: "URI"
              title: "Open Face Management"
              uri: "/face-recognition/home"
```

4. Reload Automations in Home Assistant (**Developer Tools -> YAML -> Automations**).

---

## HACS Installation (Custom Repository)

To install or track this repository using HACS:

1. Open **HACS** in Home Assistant.
2. Click the top-right menu (3 dots) and select **Custom repositories**.
3. Enter `https://github.com/spectrelabo/ha-nest-smart-face-filter` in the Repository field.
4. Select **Integration** or **Script** as the Category and click **Add**.
5. Click **Download** to pull updates.

---

## Benchmarks & Resource Usage

Tested on Raspberry Pi 4 (4GB) and Intel NUC (x86_64):

| Metric | RPi 4 (ARM64) | x86_64 CPU (4 Cores) |
| :--- | :--- | :--- |
| **Idle RAM** | 0 MB (Not running) | 0 MB (Not running) |
| **Peak RAM (Inference)** | ~185 MB | ~160 MB |
| **Detection Speed** | ~420 ms | ~110 ms |
| **Embedding Match Time** | < 5 ms | < 2 ms |

---

## Legal & Privacy Disclaimer

READ THIS: I AM NOT A LAWYER.

- **Biometrics & Local Laws:** Storing facial embeddings or capturing footage of shared spaces (e.g.,24/7 hallway, apartment corridors, public walkways) may violate privacy laws such as GDPR in Europe or local surveillance regulations. You are solely responsible for how you configure, position, and operate your cameras and data storage.
- **Liability:** Under no circumstances shall the author be held liable for legal claims, disputes with neighbors, or fines arising from the use of this software.
- **Local-First Architecture:** No image or face data is sent to cloud servers. Face images are processed locally and converted to 512-dimensional numerical vectors (`.npy`). Raw images can be discarded after processing.

---

## Support the Project

If this project saved your sanity from constant notification noise, consider supporting:

[![PayPal](https://img.shields.io/badge/Donate-PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/alessiopicco)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/blueskies360)

---
*Built for 100% local privacy, zero bloat, and peace of mind.*
