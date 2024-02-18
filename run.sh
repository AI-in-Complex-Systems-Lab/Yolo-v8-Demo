#!/bin/bash
while true; do
    source ./venv/bin/activate
    python3 main.py
    echo "Reconnecting..."
done
