# Enhancing the Decoding Rates of BATS Codes by Learning with Guided Information
This repo is for the paper [1] accepted by ISIT2022. Please refer to the accompanying paper [1] for detailed descriptions.

# Requirement
- python >= 3.7
- galois ~= 0.0.21
- torch ~= 1.10.0
- GPU support optional

# Quick Start

```
python train.py --n_pkt 64 --n_batch 10 --batch_size 8 --num_hops 10 --loss_rate 0.1
```

trained weights available upon request. contact: Jack Qing (jqing@ie.cuhk.edu.hk)

# Reference
[1] J. Qing, H. H. F. Yin, and R. W. Yeung, “Enhancing the Decoding Rates of BATS Codes by Learning with Guided Information” in Proc. ISIT’22, Espoo, Finland, Jun. 2022.