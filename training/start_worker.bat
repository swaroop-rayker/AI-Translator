@echo off
echo Starting local GPU Training Worker on Windows Host...
set PYTHONPATH=%~dp0..
set REDIS_URL=redis://localhost:6379/0
set DATA_DIR=%~dp0..\data
celery -A workers.tasks.celery_app worker --loglevel=info -Q training -c 1 --pool=solo
pause
