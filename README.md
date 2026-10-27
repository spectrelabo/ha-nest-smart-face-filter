# ha-nest-smart-face-filter

> Kill the Notification Fatigue. Event-driven, ultra-lightweight (<200MB RAM) face recognition for Google Nest & Home Assistant. Stop getting alerted every time your neighbour walks by.

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/default)
[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=spectrelabo&repository=ha-nest-smart-face-filter&category=integration)
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
+-----------------+   1. Person/Motion/Bell Event   +------------------------+
|  Google Nest    | -------------------------------> |    Home Assistant     |
|  Doorbell       |                                   | (device trigger,      |
+-----------------+                                   |  3s debounce delay)   |
                                                       +-----------+------------+
                                                                   | 2. analyze_event
                                                                   v
+-----------------+   4a. Known face                +------------------------+
| Silence / Log    | <------------------------------ |  Face Recognition      |
| (known_person_    |                                 |  Add-on (Flask/       |
|  detected event)  |                                 |  InsightFace ONNX)    |
+-----------------+                                   +-----------+------------+
                                                                   | 4b. Unknown face:
                                                                   |    - archive JPEG in
                                                                   |      unknown_faces/
                                                                   |    - fire event
                                                                   v
+---------------------+  5. Actionable push   +------------------------+
|  unknown_person_     | <-------------------- |  HA automation:         |
|  detected event      |                        |  notify_on_unknown_...  |
+---------------------+                        +-----------+------------+
                                                            |
                                                            | 6. Tap notification
                                                            v
+------------------------------------------------------------------+
|  "Gestione Volti" Dashboard (Lovelace)                             |
|  sensor.<slug>_volti_in_attesa polls GET /unknown_faces/queue      |
|  every 60s -> shows current pending face + Conferma/Scarta/Prossimo|
+------------------------------------------------------------------+
```

**Why isn't a simple `camera.snapshot` + comparison enough?** Because your
Nest takes 2-3 seconds to wake up and provide the actual event clip — that's
why the trigger applies a small `delay` before calling
`analyze_event` (which downloads the clip via `/api/nest/event_media` instead
of the live snapshot, which is almost always black/useless on Nest cameras).

**The FIFO queue ("basket") for unknown faces.** Every detected unknown face
is stored as `unknown_<timestamp>.jpg` in `unknown_faces/`. The integration
sensor polls this queue every 60 seconds (no automation is needed to "turn on"
a boolean): if you missed the push notification, or ignored it by mistake, the
face stays there waiting — the "Face Management" tab works exactly like a
basket for anything you have not handled yet, and always shows the oldest
unhandled face (FIFO). "Discard" and "Next" remove/advance it immediately,
while "Confirm" stores the embedding and automatically moves to the next one.

**Storage & retention policy.** To avoid filling your SD card with thousands
of photos of strangers/postmen/random cats, both `unknown_faces/` and
`unknown_faces/learned/` are capped at `MAX_SAVED_FACES` files (default 30,
configurable through an add-on environment variable) — above that number, the
oldest files are automatically deleted on every new save. Learned embeddings
(`.npy`, a few hundred bytes each) instead stay forever in `embeddings/`,
because they are the only thing the recognizer actually needs — there is no
reason to throw them away.

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

## Installation

### 1. Add this repository as an Add-on repository

In Home Assistant: Settings → Add-ons → Add-on Store → top-right menu (⋮)
→ Repositories → paste:

```
https://github.com/spectrelabo/ha-nest-smart-face-filter
```

### 2. Install the "Face Recognition" add-on

Search for "Face Recognition" in the Add-on Store (Local Add-ons
section), install it, and start it. Check the add-on logs and make sure they
show "Starting Face Recognition add-on on port 8080...".

### 3. Install the HACS integration

Add this same repository as a custom HACS repository
(category: Integration), install "HA Nest Smart Face Filter", restart
Home Assistant, then add the integration from Settings → Devices & Services
→ Add Integration.

If the "Addon Host" field shown during setup does not match your add-on's
actual hostname, you can check it in Settings → Add-ons → Face Recognition
→ Info → Hostname, then update it from the integration options.

### 4. Configure the automations

Copy the sample automations from `ha_configuration_snippet.yaml` into your
`configuration.yaml` (or through the Automations editor), adapting the entity
references for your Nest cameras.

### 5. Configure the "Face Management" dashboard (unknown-face queue)

Copy `lovelace_face_validation_card.yaml` into a view of your dashboard
(YAML mode: dashboard → ⋮ → Edit dashboard → ⋮ → Edit in YAML).
Remember that HA expects a `views:` array — if this is your only view, wrap
the file contents with `views:` followed by a dash `- title: ...` before
pasting, otherwise HA will complain with "Expected an array value".

⚠️ **Heads-up if you customize the card:** the native Lovelace
`type: picture` card **does not evaluate Jinja templates** in the `image:`
field — if you try to build the URL dynamically there, the browser will
receive the literal template string and you will never see the photo. That is
why the photo card uses `type: markdown` with an inline `<img>` tag, which
actually supports templates.

### 6. Add the missing script and helper ("Confirm" workflow)

The "Confirm" button on the dashboard calls `script.conferma_gestione_volto`
and reads the typed name from `input_text.nome_nuovo_volto`. **Neither exists
by default in Home Assistant** — you must add them manually by copying the
contents of `ha_backend_logic.yaml` (script) and section 1 of
`ha_configuration_snippet.yaml` (helper) into your `configuration.yaml` (or
into the corresponding `scripts.yaml`/`input_text.yaml` files if you use
`!include`). "Discard" and "Next" already work immediately because they call
services registered directly by the integration — useful, but if you forget the
script, "Confirm" will appear to "work" (no visible error) while never
actually registering the face.

### 7. Allow the unknown-faces directory (`allowlist_external_dirs`)

The `learn_face` service validates every path with Home Assistant's
`is_allowed_path()` check. The face queue lives under
`/config/face_recognition/unknown_faces/` by default, which HA does **not**
allow by default. Add the following block to `configuration.yaml` and restart
Home Assistant:

```yaml
homeassistant:
  allowlist_external_dirs:
    - /config/face_recognition/unknown_faces
```

Without this step, pressing "Confirm" will raise a `ServiceValidationError`
and the face will not be registered.

After pasting, **restart Home Assistant** (mandatory if you add new
top-level keys like `input_text:`/`script:` directly in `configuration.yaml`;
if you already use `!include scripts.yaml`, a "Reload Scripts" from the
Automations & Scenes menu is enough).

### 7. Branding: literally put a face on the project

Want the add-on and integration to stop showing the generic gray icon?
You need two PNGs per component:

**For the add-on** (`addon/face_recognition/`):
```
addon/face_recognition/
├── config.yaml
├── icon.png   # 128x128px, quadrato 1:1
└── logo.png   # 250x100px (aspect ratio libero)
```
Just place them at the root of the add-on folder, next to
`config.yaml` — no extra configuration required, the Supervisor will detect
them automatically on the next update.

**For the HACS integration** (`custom_components/ha_nest_smart_face_filter/`):
```
custom_components/ha_nest_smart_face_filter/
├── manifest.json
└── brand/
    ├── icon.png        # 128x128px, quadrato 1:1
    ├── logo.png        # 250x100px
    ├── dark_icon.png   # opzionale, variante dark mode
    └── logo@2x.png     # opzionale, variante HiDPI
```
Custom integrations can now include their own branding images without
opening a PR against the central
[`home-assistant/brands`](https://github.com/home-assistant/brands) repository
— HACS and the frontend pull them directly from here. If the brands repository
still does not recognize the domain, the icon stays the default one until you
add the `brand/` folder: not a bug, just cache laziness.

---

## FAQ

**The "Face Management" dashboard stays empty even though I know an
unknown face was detected.**
Check that the `sensor.<slug>_volti_in_attesa` sensor exists and is not
`unavailable`. If your integration is still stuck on the old
`input_boolean.volto_in_attesa` architecture, note that **nothing ever turns
it on** — not the add-on (which only fires events, it does not change states)
nor the sample automations. Update to v2.2.0+ (sensor polling the queue
directly) to remove this issue at the root.

**I pressed "Confirm" but the face does not disappear from the queue and I
can't find the embedding.**
See point 6 above: you are almost certainly missing
`script.conferma_gestione_volto` or `input_text.nome_nuovo_volto`. Verify in
Settings → Automations & Scenes → Scripts that the script really exists.

**The photo in the card is not visible / broken icon.**
From v2.4.0 the face preview is served through an authenticated HA
`image` entity (`image.<slug>_pending_face`). The Lovelace dashboard
uses a `picture-entity` card — make sure the `entity:` field matches
the actual entity_id shown in Settings → Devices & Services →
your integration → Entities.

**I tried opening `/local/...` manually and the file no longer exists.**
Since v2.4.0 no face images are written to `/config/www/`. The pending
face is served exclusively through the authenticated image entity
(`/api/image_proxy/<entity_id>`), so it is no longer accessible without
a valid Home Assistant session. This is intentional.

**HACS does not find the update to a new version.**
Try a forced refresh: HACS → your repository → menu ⋮ → "Redownload".
Sometimes the HACS cache takes a bit to notice new releases.

**Why does Nest take 2-3 seconds to "wake up"?**
That is a Nest device/API limitation, not ours — the trigger adds a short
`delay` before calling `analyze_event` so the real event clip (not a black
snapshot) is already available when we request it.

**If I miss/ignore the notification, is the unknown face gone forever?**
No: it stays in the FIFO queue (the "basket") until you handle it — the Face
Management tab will always show the oldest not-yet-handled one whenever you
open it, even days later (until the retention limit, see above).

**Can I use multiple Nest cameras with the same pipeline?**
Yes, just duplicate the `device` triggers in the pipeline automation for each
extra `device_id`; the add-on and the queue are already shared, so no separate
instance is needed.

---

## Known Issues / Troubleshooting

- **Crash when opening the integration Options** (`AttributeError: property
  'config_entry' ... has no setter`): fixed bug — on Home Assistant
  2024.11+, `OptionsFlow.config_entry` became a core-managed read-only
  property; code versions before this fix assigned `self.config_entry`
  manually in `__init__`, causing the crash. Update to the latest integration
  version.
- **Up to 60s delay in dashboard updates**: the sensor polls the queue
  every 60 seconds to avoid hammering the add-on. After Confirm/Discard/Next
  the refresh is still immediate (direct coordinator call); the delay is only
  noticeable for a newly detected face that arrives passively.
- **Lovelace preview uses authenticated image entity**: from v2.4.0 the
  face preview card uses `picture-entity` pointing at
  `image.<slug>_pending_face` instead of an unauthenticated `/local/` URL.
  If the card shows a broken icon, verify that the entity_id in
  `lovelace_face_validation_card.yaml` matches your instance's actual
  entity_id (Settings → Devices & Services → your integration → Entities).
- **`learn_face` requires `allowlist_external_dirs`**: from v2.4.0 the
  service validates every path with `is_allowed_path()`. Add
  `/config/face_recognition/unknown_faces` to `allowlist_external_dirs`
  in `configuration.yaml` (see Setup step 7 above) or "Confirm" will
  raise a `ServiceValidationError`.
- **`camera.snapshot` on Nest often returns black frames**: that is why
  the pipeline uses `analyze_event` (real event clip via
  `/api/nest/event_media`) instead of the live snapshot.

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
