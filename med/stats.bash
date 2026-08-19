#!/bin/bash

DIR=${1}

for filename in ${DIR}/experiment-lats-*2.5*.json
do
  python3 qa/stats.py --output_file_name "${filename}"
  echo ""
done
