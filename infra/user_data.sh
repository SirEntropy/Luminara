#!/bin/bash
set -e
# Install Python & dependencies
apt-get update
apt-get install -y python3-pip git
cd /home/ubuntu || cd /root
if [ ! -d "Luminara" ]; then
  git clone https://github.com/SirEntropy/Luminara.git
fi
cd Luminara
pip3 install --upgrade pip
pip3 install fastapi gunicorn uvicorn
pip3 install -r requirements.txt
# Launch FastAPI with gunicorn+uvicorn
nohup gunicorn -k uvicorn.workers.UvicornWorker src.api:app --bind 0.0.0.0:80 --workers 2 &
