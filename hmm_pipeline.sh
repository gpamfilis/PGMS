python step_1_elo_calculation.py
python step_2_elo_get_hidden_get_observed.py $1 $2 $3
python step_3_compute_px_pxx_pyx.py $1 $2 $3
python step_4_fb.py $1 $3
