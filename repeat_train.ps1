# pretrained w/ noise
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --pretrained --add_noise --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.3
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --pretrained --add_noise --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.2
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --pretrained --add_noise --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.1
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --pretrained --add_noise --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.01
# pretrained w/o noise
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --pretrained --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.3
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --pretrained --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.2
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --pretrained --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.1
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --pretrained --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.01
# random w/ noise
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --add_noise --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.3
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --add_noise --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.2
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --add_noise --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.1
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --add_noise --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.01
# random w/o noise
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.3
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.2
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.1
python transfer_learning.py --model facebook/deit-tiny-patch16-224 --directory . --epochs 4.0 --batch_size 32 --learning_rate 1e-3 --weight_decay 0.01