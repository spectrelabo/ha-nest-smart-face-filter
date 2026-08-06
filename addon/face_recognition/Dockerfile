ARG BUILD_FROM
FROM $BUILD_FROM

# Dipendenze native richieste da opencv-python-headless e Python stesso
# (immagine base-debian non include Python preinstallato, a differenza di base-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# --break-system-packages: sicuro qui perché il container è isolato e dedicato
# a questo add-on (non è un sistema Python condiviso) — bypassa la protezione
# PEP 668 di Debian bookworm.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY recognition.py analyze_server.py ./

COPY run.sh /
RUN chmod a+x /run.sh

# NOTA: EXPOSE è puramente documentale (non pubblica la porta sull'host).
# L'isolamento di rete reale è imposto da config.yaml ("ports: 8080/tcp: null"),
# che espone la porta solo sulla rete Docker interna di Home Assistant.
EXPOSE 8080

CMD [ "/run.sh" ]
