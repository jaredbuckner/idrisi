#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/python/src"
cd "${REPO_ROOT}"

python -midrisi.cs \
       --separate 29 \
       --river_source_grade 0.1:3.5 \
       --river_min_grade 0.0 \
       --river_separation 700:1800 \
       --river_segment_length 400:800 \
       --flood_plain_grade 0.5:2.5 \
       --flood_plain_width 150:650 \
       --foothill_grade 1:15 \
       --foothill_width 25:325 \
       --midhill_grade 1:15 \
       --midhill_width 25:325 \
       --shoulder_grade 1:30 \
       --shoulder_width 50:350 \
       --shore_width 0:1000 \
       --shore_grade 0.5:5 \
       --peak_grade 15:60 \
       --variance_scale 2 \
       --wrinkle_mean_length 300 \
       --wrinkle_dev_length 100 \
       --wrinkle_grade 5:15


#       --separate 43
#       --wrinkle_mean_length 300 \
#       --wrinkle_dev_length 100 \
#       --wrinkle_grade 10:15 \
       
