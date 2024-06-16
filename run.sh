#!/bin/bash

for file in *.py; do
    if [ "$file" != "gpu.py" ]; then
        CUDA_VISIBLE_DEVICES="6,7" python "$file"
    fi
done