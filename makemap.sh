#!/bin/bash

export PYTHONPATH=src

python -midrisi.cs \
       --separate 43 \
       --river_source_grade 5 \
       --river_min_grade 0.0 \
       --river_separation 1100 \
       --river_segment_length 180 \
       --shore_grade 2 \
       --flood_plain_grade 1 \
       --flood_plain_width 100 \
       --foothill_width 250 \
       --foothill_grade 5 \
       --midhill_width 90 \
       --midhill_grade 10 \
       --shoulder_width 90 \
       --shoulder_grade 35 \
       --peak_grade 70 \
       --wrinkle_mean_length 0 \
       --wrinkle_dev_length 0

# --separate 43
