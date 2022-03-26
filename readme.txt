main training files:
DeePSLoc:
patch-branch: train_patch_attn.py
downsample_branch: train_resample_attn.py
D+P: train_merge.py

DeePSLoc_multi:
patch-branch: multi_train_attn.py --patch True
downsample_branch: multi_train_attn.py --patch False
D+P: multi_train_merge.py

Model file：
models/triplet_transformer.py
