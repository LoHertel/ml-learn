#!/bin/sh

# exit script immediately on non-zero status
set -e

. /venv/bin/activate

exec gunicorn --bind 0.0.0.0:9696 --forwarded-allow-ips='*' predict:app