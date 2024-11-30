#!/bin/bash

DIR=${1}

for filename in ${DIR}/experiment-lats-*.json
do
  python3 qa/stats.py --output_file_name "${filename}"
  echo ""
done
