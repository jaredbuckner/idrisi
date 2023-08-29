#!/bin/bash

export PYTHONPATH=src

python -midrisi.cs \
       --separate 43 \
       --river_source_grade 1 \
       --river_min_grade 0.0 \
       --river_separation 1800 \
       --river_segment_length 350 \
       --shore_grade 2 \
       --flood_plain_grade 0.5 \
       --flood_plain_width 350 \
       --foothill_width 120 \
       --foothill_grade 10 \
       --midhill_width 180 \
       --midhill_grade 20 \
       --shoulder_width 90 \
       --shoulder_grade 35 \
       --peak_grade 50
