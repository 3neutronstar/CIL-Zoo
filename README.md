# Class Incremental Learning Zoo - pytorch version
Unofficial pytorch version of reproducing the method of class incremental learning (Working) in classification

Currently only CIFAR100 dataset is available.

- How to Implement (iCaRL)
```
    python main.py train --train-mode icarl --gpu-ids 0 --model resnet32
```

- How to Implement (EEIL, with BiC setting)
```
    python main.py train --train-mode eeil --gpu-ids 2 --task-size 10 --model resnet32 --batch-size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr-steps 100,150,200 --weight-decay 0.0002
```

- How to Implement (EEIL, with EEIL setting)
```
    python main.py train --train-mode eeil --gpu-ids 2 --task-size 10 --model resnet32 --batch-size 128 --lr 0.1 --gamma 0.1 --epochs 40 --lr-steps 10,20,30 --weight-decay 0.0001
```

- How to Implement (BiC)
```
    python main.py train --train-mode bic --gpu-ids 2 --task-size 10 --model resnet32 --batch-size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr-steps 100,150,200 --weight-decay 0.0002
```

- 5 task (Top-1 Accuracy)

| 5 task |    1 |     2 |     3 |     4 |     5 |
|--------|-----:|------:|------:|------:|------:|
| iCaRL  | 78.35| 65.25 | 54.15 | 47.94 | 38.7  |
| EEIL   | 82.5 | 72.12 | 65.75 | 59.08 | 54.23 |
| BiC    | 81.15| 71.43 | 64.28 | 58.56 | 54.91 |

EEIL use BiC settings for reproducing, we provide EEIL settings also.</br>
- 10 task

| 10 task |  1 |    2 |    3 |    4 |     5 | 6     | 7     | 8     | 9     | 10    |
|---------|---:|-----:|-----:|-----:|------:|-------|-------|-------|-------|-------|
| iCaRL   |80.6| 66.7 | 59.67|52.88 | 48.36 | 44.68 | 41.76 | 38.19 | 36.03 | 33.19 |
| EEIL    | 87 | 77.7 | 73.7 | 66.5 | 61.66 | 58.37 | 55.27 | 51.21 | 48.59 | 45.47 |
| BiC     |87.7| 80.4 | 72.17| 66.45 | 61.92 | 57.78 | 54.61 | 51.4 | 50.28 | 47.78 |
- 20 task

| 20 task |    1 |    2 |    3 |     4 |     5 | 6    | 7     | 8    | 9     | 10   | 11    | 12    | 13    | 14    | 15    | 16    | 17    | 18    | 19    | 20    |
|---------|-----:|-----:|-----:|------:|------:|------|-------|------|-------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| iCaRL   | 79.6	|73.2	|67.8|	63.35	|63.92	|58.93|	57.6|	53.9|	50.51|	49	|47.22|	42.97|	41.46|	38.04|	32.43|	30.28|	27.13|	25.53	|23.06|	20|
| EEIL    | 84.4 | 83.3 | 75.4 | 69.65 | 70.36 | 65.6 | 63.86 | 56.8 | 54.29 | 52.3 | 50.42 | 48.32 | 47.95 | 45.39 | 43.12 | 42.81 | 40.58 | 40.46 | 39.65 | 36.12 |
| BiC     | 87.4 | 85.4 | 78.47| 76.4|73.84|68.87|65.54|61.68|57.67|54.8|53|50.68|51.21|49.62|48.77|45.76|45.29|43.67|41.27|39.64|


### Reference
- iCaRL construct exemplar set is borrowed from [iCaRL-pytorch](https://github.com/DRSAD/iCaRL/blob/master/iCaRL.py).
