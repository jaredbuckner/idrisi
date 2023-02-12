#!/bin/bash

export PYTHONPATH=src

python -midrisi.cs \
       --separate 43 \
       --river_source_grade 10 \
       --river_min_grade 0.1 \
       --river_separation 1800 \
       --river_segment_length 350 \
       --shore_grade 10 \
       --flood_plain_grade 1 \
       --flood_plain_width 350 \
       --foothill_width 120 \
       --foothill_grade 5 \
       --midhill_width 180 \
       --midhill_grade 10 \
       --shoulder_width 90 \
       --shoulder_grade 20 \
       --peak_grade 45
