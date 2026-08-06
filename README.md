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

## Installazione

### 1. Aggiungi questo repository come repository di Add-on

In Home Assistant: Impostazioni → Add-on → Add-on Store → menu (⋮) in alto
a destra → Repository → incolla:

```
https://github.com/spectrelabo/ha-nest-smart-face-filter
```

### 2. Installa l'Add-on "Face Recognition"

Cerca "Face Recognition" nello store degli Add-on (sezione Local Add-ons),
installalo e avvialo. Verifica nei log dell'add-on che riporti
"Avvio Face Recognition add-on su porta 8080...".

### 3. Installa l'integrazione HACS

Aggiungi questo stesso repository come repository HACS personalizzato
(categoria Integration), installa "HA Nest Smart Face Filter", riavvia
Home Assistant, poi aggiungi l'integrazione da Impostazioni → Dispositivi e
Servizi → Aggiungi integrazione.

Se il campo "Addon Host" mostrato durante la configurazione non corrisponde
all'hostname reale del tuo add-on, puoi verificarlo in Impostazioni →
Add-on → Face Recognition → Informazioni → Hostname, e aggiornarlo dalle
opzioni dell'integrazione.

### 4. Configura le automazioni

Copia le automazioni di esempio da `ha_configuration_snippet.yaml` nel tuo
`configuration.yaml` (o tramite l'editor Automazioni), adattando le entity
delle tue telecamere Nest.

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
