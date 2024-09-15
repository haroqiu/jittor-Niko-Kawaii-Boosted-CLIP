#!/bin/bash

adapters=(mlp attn conv)
epochs=(50 60 70 80 90)
for adapter in ${adapters[@]}; do
    for epoch in ${epochs[@]}; do
        CUDA_VISIBLE_DEVICES=1 python -u adapter_boosting.py --adapter_type $adapter --num_epochs $epoch --num_adapters=1 --eval True > single_${adapter}_${epoch}.log
    done
done