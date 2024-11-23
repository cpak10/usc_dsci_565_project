#!/bin/zsh

#python3 transfer_learning.py --model='google/efficientnet-b3' --train_test_split=0.25 --learning_rate=1e-4 --weight_decay=0 --batch_size=16 --epochs=3
python3 transfer_learning.py --model='google/efficientnet-b3' --train_test_split=0.25 --learning_rate=1e-3 --weight_decay=0.1 --batch_size=32 --epochs=6
python3 transfer_learning.py --model='google/efficientnet-b3' --train_test_split=0.25 --learning_rate=1e-3 --weight_decay=0.2 --batch_size=32 --epochs=6
python3 transfer_learning.py --model='google/efficientnet-b3' --train_test_split=0.25 --learning_rate=1e-3 --weight_decay=0.3 --batch_size=32 --epochs=6
python3 transfer_learning.py --model='google/efficientnet-b3' --train_test_split=0.25 --learning_rate=1e-3 --weight_decay=0.1 --batch_size=32 --epochs=6 --add_noise
python3 transfer_learning.py --model='google/efficientnet-b3' --train_test_split=0.25 --learning_rate=1e-3 --weight_decay=0.2 --batch_size=32 --epochs=6 --add_noise
python3 transfer_learning.py --model='google/efficientnet-b3' --train_test_split=0.25 --learning_rate=1e-3 --weight_decay=0.3 --batch_size=32 --epochs=6 --add_noise

