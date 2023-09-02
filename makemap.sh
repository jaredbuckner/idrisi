#!/bin/bash

export PYTHONPATH=src

python -midrisi.cs \
       --separate 29 \
       --river_source_grade 7 \
       --river_min_grade 0.0 \
       --river_separation 1100 \
       --river_segment_length 180 \
       --shore_grade 4 \
       --flood_plain_grade 1 \
       --flood_plain_width 50 \
       --foothill_width 75 \
       --foothill_grade 5 \
       --midhill_width 95 \
       --midhill_grade 10 \
       --shoulder_width 90 \
       --shoulder_grade 35 \
       --peak_grade 70 \
       --wrinkle_mean_length 0 \
       --wrinkle_dev_length 0

# --separate 43
