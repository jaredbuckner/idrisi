#!/bin/bash

export PYTHONPATH=src

python -midrisi.cs \
       --separate 29 \
       --river_source_grade 0.1:10 \
       --river_min_grade 0.0 \
       --river_separation 900:1800 \
       --river_segment_length 400:800 \
       --flood_plain_grade 0.5:2.5 \
       --flood_plain_width 50:550 \
       --foothill_grade 10:20 \
       --foothill_width 50:350 \
       --midhill_grade 0.5:8 \
       --midhill_width 50:350 \
       --shoulder_grade 0.5:30 \
       --shoulder_width 50:350 \
       --shore_width 0:1000 \
       --shore_grade 0.5:60 \
       --peak_grade 45:60 \
       --wrinkle_mean_length 0 \
       --wrinkle_dev_length 0

# --separate 43
       
