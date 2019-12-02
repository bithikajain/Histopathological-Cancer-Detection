#!/bin/bash
python scripts/train_and_test_no_early_stop.py --resuming \
  --run-name no_early_stop \
  --checkpoint-file-path /home/prayush/bj/pcam/BEST_MODEL_TRAINED_ON_LAPTOP/saved-model-04-0.931.hdf5 \
  --verbose
