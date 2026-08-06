#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Avvio Face Recognition add-on su porta 8080..."

exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 120 analyze_server:app
