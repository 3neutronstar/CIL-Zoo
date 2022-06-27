# CIL-Zoo
Class Incremental Learning

## Preparation
- Requirements

## Execution Command
``` shell scripts
    # iCaRL
    python main.py train --train_mode icarl --model resnet32 --dataset cifar100 --batch_size 128 --lr 2 --lr_steps 49,63 --epochs 70 --task_size 5 --weight_decay 1e-5 --gamma 0.2 --memory_size 2000
    # EEIL
    python main.py train --train_mode eeil --model resnet32 --dataset cifar100 --batch_size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr_steps 100,150,200 --weight_decay 0.0002
```
