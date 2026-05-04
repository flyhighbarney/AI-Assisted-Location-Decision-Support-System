#!/bin/bash
set -e

apt-get update
apt-get install -y curl gnupg apt-transport-https unixodbc unixodbc-dev

curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18

python -m pip install --upgrade pip
pip install -r requirements.txt

exec gunicorn --bind=0.0.0.0:${PORT:-8000} app:app
