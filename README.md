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

**Perché non basta un semplice `camera.snapshot` + confronto?** Perché il tuo
Nest ci mette 2-3 secondi a svegliarsi e mandare il clip evento reale — ecco
perché il trigger applica un piccolo `delay` prima di chiamare
`analyze_event` (che scarica il clip via `/api/nest/event_media` invece dello
snapshot live, quasi sempre nero/inutilizzabile sulle Nest).

**La coda FIFO ("basket") dei volti sconosciuti.** Ogni volto sconosciuto
rilevato viene archiviato come `unknown_<timestamp>.jpg` in
`unknown_faces/`. Il sensore dell'integrazione fa polling di questa coda ogni
60 secondi (non serve nessuna automazione che "accenda" un booleano): se hai
perso la notifica push, o l'hai ignorata per sbaglio, il volto resta lì in
attesa — la tab "Gestione Volti" fa esattamente da cestino/basket per tutto
ciò che non hai ancora gestito, e mostra sempre il più vecchio non gestito
(FIFO). "Scarta" e "Prossimo" lo rimuovono/avanzano subito, "Conferma"
registra l'embedding e passa al successivo automaticamente.

**Storage & retention policy.** Per non riempire la SD card con migliaia di
foto di sconosciuti/postini/gatti di passaggio, sia `unknown_faces/` sia
`unknown_faces/learned/` sono cappate a `MAX_SAVED_FACES` file (default 30,
configurabile via variabile d'ambiente dell'add-on) — oltre quel numero, i
file più vecchi vengono eliminati automaticamente ad ogni nuovo salvataggio.
Gli embedding appresi (`.npy`, poche centinaia di byte l'uno) restano invece
per sempre in `embeddings/`, perché sono l'unica cosa che serve davvero al
riconoscimento — non c'è motivo di buttarli via.

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

### 5. Configura la dashboard "Gestione Volti" (coda dei volti sconosciuti)

Copia `lovelace_face_validation_card.yaml` in una view della tua dashboard
(modalità YAML: dashboard → ⋮ → Modifica dashboard → ⋮ → Modifica in YAML).
Ricordati che serve un array `views:` — se questa è l'unica view, avvolgi il
contenuto del file con `views:` seguito da un trattino `- title: ...` prima
di incollarlo, altrimenti HA si lamenta con "Expected an array value".

⚠️ **Attenzione se personalizzi la card:** la card nativa Lovelace
`type: picture` **non valuta i template Jinja** nel campo `image:` — se
provi a costruire l'URL dinamicamente lì dentro, il browser riceverà la
stringa del template alla lettera e non vedrai mai la foto. Per questo la
card foto usa `type: markdown` con un tag `<img>` inline, che i template li
supporta per davvero.

### 6. Aggiungi lo script e l'helper mancanti (workflow "Conferma")

Il tasto "Conferma" nella dashboard chiama `script.conferma_gestione_volto`
e legge il nome digitato da `input_text.nome_nuovo_volto`. **Nessuno dei due
esiste di default in Home Assistant** — vanno aggiunti a mano copiando il
contenuto di `ha_backend_logic.yaml` (script) e la sezione 1 di
`ha_configuration_snippet.yaml` (helper) nel tuo `configuration.yaml` (o nei
rispettivi `scripts.yaml`/`input_text.yaml`, se usi `!include`). "Scarta" e
"Prossimo" invece funzionano già subito, perché chiamano servizi registrati
direttamente dall'integrazione — non ti bastano ma se ti scordi lo script,
"Conferma" sembrerà "funzionare" (nessun errore visibile) ma non registrerà
mai nessun volto.

Dopo aver incollato, **riavvia Home Assistant** (obbligatorio se aggiungi
nuove chiavi di primo livello come `input_text:`/`script:` direttamente in
`configuration.yaml`; se usi già `!include scripts.yaml`, un "Ricarica
script" dal menu Automazioni e scene basta).

### 7. Branding: metti la faccia (letteralmente) al progetto

Vuoi che l'add-on e l'integrazione smettano di avere l'iconcina generica
grigia? Servono due PNG per componente:

**Per l'add-on** (`addon/face_recognition/`):
```
addon/face_recognition/
├── config.yaml
├── icon.png   # 128x128px, quadrato 1:1
└── logo.png   # 250x100px (aspect ratio libero)
```
Basta metterli alla radice della cartella dell'add-on, accanto a
`config.yaml` — nessuna configurazione aggiuntiva richiesta, il Supervisor
li rileva da solo al prossimo aggiornamento.

**Per l'integrazione HACS** (`custom_components/ha_nest_smart_face_filter/`):
```
custom_components/ha_nest_smart_face_filter/
├── manifest.json
└── brand/
    ├── icon.png        # 128x128px, quadrato 1:1
    ├── logo.png        # 250x100px
    ├── dark_icon.png   # opzionale, variante dark mode
    └── logo@2x.png     # opzionale, variante HiDPI
```
Le integrazioni custom possono ormai includere le proprie immagini di brand
senza dover fare una PR al repository centrale
[`home-assistant/brands`](https://github.com/home-assistant/brands) — HACS e
il frontend le pescano direttamente da qui. Se il repository brands non
riconosce ancora il dominio, l'icona resta quella di default finché non
aggiungi la cartella `brand/`: non è un bug, è solo pigrizia del cache.

---

## FAQ

**La dashboard "Gestione Volti" resta vuota anche se so che è stato rilevato
un volto sconosciuto.**
Controlla che il sensore `sensor.<slug>_volti_in_attesa` esista e non sia
`unavailable`. Se l'integrazione è ferma alla vecchia architettura basata su
`input_boolean.volto_in_attesa`, sappi che **niente lo accende mai** — né
l'add-on (che spara solo eventi, non tocca stati) né le automazioni di
esempio. Aggiorna alla v2.2.0+ (sensore che fa polling diretto della coda)
per eliminare questo problema alla radice.

**Ho premuto "Conferma" ma il volto non sparisce dalla coda e non trovo
l'embedding.**
Vedi punto 6 sopra: quasi certamente ti manca `script.conferma_gestione_volto`
o `input_text.nome_nuovo_volto`. Verifica in Impostazioni → Automazioni e
scene → Script che lo script esista davvero.

**La foto nella card non si vede / icona rotta.**
Se hai personalizzato la card con `type: picture` e un URL templated, non
funzionerà mai (vedi punto 5). Usa la card `markdown` fornita nel repo.

**Ho provato ad aprire `/local/...` a mano e mi dà timeout, non 404.**
Controlla se stai passando da un reverse proxy/dominio esterno che inoltra
solo la porta 443 — in quel caso l'URL "interno" con `:8123` esplicito non è
raggiungibile dall'esterno. Usa lo stesso indirizzo (senza porta) che usi
normalmente per aprire la dashboard.

**HACS non trova l'aggiornamento a una nuova versione.**
Prova un refresh forzato: HACS → il tuo repository → menu ⋮ → "Ridownload".
A volte la cache di HACS ci mette un po' a notare le nuove release.

**Perché il Nest ci mette 2-3 secondi a "svegliarsi"?**
È un limite del dispositivo/API Nest, non nostro — per questo il trigger
applica un breve `delay` prima di chiamare `analyze_event`, così il clip
evento reale (non uno snapshot nero) è già disponibile quando lo richiediamo.

**Se perdo/ignoro la notifica, il volto sconosciuto è perso per sempre?**
No: resta nella coda FIFO (il "basket") finché non lo gestisci — la tab
Gestione Volti mostrerà sempre il più vecchio non ancora gestito ad ogni
apertura, anche giorni dopo (fino al limite di retention, vedi sopra).

**Posso usare più telecamere Nest con la stessa pipeline?**
Sì, basta duplicare i trigger `device` nell'automazione della pipeline per
ogni `device_id` aggiuntivo; l'add-on e la coda sono già condivisi e non
serve nessuna istanza separata.

---

## Known Issues / Troubleshooting

- **Crash aprendo le Opzioni dell'integrazione** (`AttributeError: property
  'config_entry' ... has no setter`): bug corretto — su Home Assistant
  2024.11+, `OptionsFlow.config_entry` è diventata una property read-only
  gestita dal core; versioni del codice precedenti a questa correzione
  assegnavano `self.config_entry` manualmente in `__init__`, causando il
  crash. Aggiorna all'ultima versione dell'integrazione.
- **Ritardo fino a 60s nell'aggiornamento della dashboard**: il sensore fa
  polling della coda ogni 60 secondi per non martellare l'add-on. Dopo
  Conferma/Scarta/Prossimo il refresh è comunque immediato (chiamata diretta
  al coordinator), il ritardo si nota solo per un volto nuovo rilevato
  passivamente.
- **`type: picture` di Lovelace non supporta template Jinja**: limite nativo
  di Home Assistant, non un nostro bug — usa `markdown` con `<img>` come
  fatto nel file fornito, oppure una custom card come `html-template-card`
  se ti serve qualcosa di più elaborato.
- **`camera.snapshot` su Nest spesso restituisce frame neri**: per questo la
  pipeline usa `analyze_event` (clip evento reale via
  `/api/nest/event_media`) invece dello snapshot live.

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
