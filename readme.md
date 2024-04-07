Pipeline:

1) move folders with images to Dataset/ folder

2) crop them to needed size with make_crops.py -c 256 for kbnet or with make_crops_not_strict -c 1024 for Restormer

3) Denoise with either run_kbnet.py or run_restormer.py

4) Restore denoised images from patches with restore_iamges.py

5) Calculate metrics for whole pictures or for patches with calc_metrics.py
