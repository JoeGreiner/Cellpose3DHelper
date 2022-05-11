#!/usr/bin/env zsh

python -m cellpose --train --use_gpu --dir data/extracted_slices_train --test_dir data/extracted_slices_test --verbose --mask_filter _masks --n_epochs 100 --pretrained_model cyto2
