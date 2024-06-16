#!/bin/bash

for file in *.py; do
    if [ "$file" != "gpu.py" ]; then
        python "$file"
    fi
done