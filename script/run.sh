# 同一为工作目录
cd ../src/
gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 eknowledge.main:app
