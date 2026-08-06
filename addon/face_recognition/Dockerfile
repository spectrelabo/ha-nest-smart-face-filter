ARG BUILD_FROM
FROM $BUILD_FROM

# Dipendenze native richieste da opencv-python-headless su Alpine/Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recognition.py analyze_server.py ./

COPY run.sh /
RUN chmod a+x /run.sh

EXPOSE 8080

CMD [ "/run.sh" ]
