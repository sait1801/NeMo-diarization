docker run --gpus all --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /teamspace/studios/this_studio/NeMo-diarization/models:/models \
  --shm-size=30g tritonn:latest \
  tritonserver --model-repository=/models



#   ----------------------NeMo--------------
  docker run --gpus all --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /teamspace/studios/this_studio/NeMo-diarization/models:/models \
  --shm-size=30g tritonn:latest \
  tritonserver --model-repository=/models