#!/bin/bash

MODEL_NAME=${1}
TEMP=${2}
CONFIG=${3}

while true
do
  PYTHONFAULTHANDLER=1
  PYDEVD_USE_CYTHON=NO
  python qa/run.py \
    --model_name=${MODEL_NAME} \
    --chain_type=lats \
    --output_file_name=experiment-lats-${MODEL_NAME}-${TEMP}.json \
    --temperature=${TEMP} \
    --config_path=${CONFIG} \
    --max_output_tokens=8192 \
    --input_file=oncology2.json \
    --sample_size=1 \
    --callback \
  && break
done
