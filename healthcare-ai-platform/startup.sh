#!/bin/bash
# Azure Web App startup script
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
