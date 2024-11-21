#!/bin/zsh

#python3 transfer_learning.py --model='google/efficientnet-b3' --pretrained=False --train_test_split=0.25 --learning_rate=1e-3 --weight_decay=0
#python3 transfer_learning.py --model='google/efficientnet-b3' --pretrained=False --train_test_split=0.25 --learning_rate=1e-2 --weight_decay=0
python3 transfer_learning.py --model='google/efficientnet-b3' --train_test_split=0.25 --learning_rate=1e-3 --weight_decay=0 --batch_size=16
