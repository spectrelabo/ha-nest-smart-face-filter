#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Starting Face Recognition add-on on port 8080..."

exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 120 analyze_server:app
