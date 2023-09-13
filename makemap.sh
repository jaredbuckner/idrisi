#!/bin/bash

export PYTHONPATH=src

python -midrisi.cs \
       --separate 29 \
       --river_source_grade 5:10 \
       --river_min_grade 0.0 \
       --river_separation 900:1800 \
       --river_segment_length 150:250 \
       --flood_plain_grade 0.5:2.5 \
       --flood_plain_width 400:900 \
       --foothill_grade 1:9 \
       --foothill_width 100:190 \
       --midhill_grade 4:16 \
       --midhill_width 80:140 \
       --shoulder_grade 10:30 \
       --shoulder_width 80:140 \
       --shore_width 0:1000 \
       --shore_grade 0:10 \
       --peak_grade 30:60 \
       --wrinkle_mean_length 0 \
       --wrinkle_dev_length 0

# --separate 43
       
