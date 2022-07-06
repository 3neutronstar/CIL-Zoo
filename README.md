# RelatedWork

- How to Implement (iCaRL)
```
    python main.py train --train-mode icarl --gpu-ids 0,1 --model resnet32
```

- How to Implement (EEIL)
```
    python main.py train --train-mode eeil --gpu-ids 2 --task-size 10 --model resnet32 --batch-size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr-steps 100,150,200 --weight-decay 0.0002
```


- How to Implement (EEIL+NI)
```
    python main.py train --train-mode eeil --gpu-ids 1 --task-size 10 --model resnet34 --batch-size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr-steps 100,150,200 --weight-decay 0.0002 --natural-inversion True --inversion_epochs 2000
``` 

- How to Implement (BiC)
```
    python main.py train --train-mode bic --gpu-ids 2 --task-size 10 --model resnet32 --batch-size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr-steps 100,150,200 --weight-decay 0.0002
```

- 5 task

| 5 task |    1 |     2 |     3 |     4 |     5 |
|--------|-----:|------:|------:|------:|------:|
| iCaRL  |      |       |       |       |       |
| EEIL   | 82.5 | 72.12 | 65.75 | 59.08 | 54.23 |
| BiC    |      |       |       |       |       |

- 10 task

| 10 task |  1 |    2 |    3 |    4 |     5 | 6     | 7     | 8     | 9     | 10    |
|---------|---:|-----:|-----:|-----:|------:|-------|-------|-------|-------|-------|
| iCaRL   |    |      |      |      |       |       |       |       |       |       |
| EEIL    | 87 | 77.7 | 73.7 | 66.5 | 61.66 | 58.37 | 55.27 | 51.21 | 48.59 | 45.47 |
| BiC     |    |      |      |      |       |       |       |       |       |       |
|         |    |      |      |      |       |       |       |       |       |       |

- 20 task

| 20 task |    1 |    2 |    3 |     4 |     5 | 6    | 7     | 8    | 9     | 10   | 11    | 12    | 13    | 14    | 15    | 16    | 17    | 18    | 19    | 20    |
|---------|-----:|-----:|-----:|------:|------:|------|-------|------|-------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| iCaRL   |      |      |      |       |       |      |       |      |       |      |       |       |       |       |       |       |       |       |       |       |
| EEIL    | 84.4 | 83.3 | 75.4 | 69.65 | 70.36 | 65.6 | 63.86 | 56.8 | 54.29 | 52.3 | 50.42 | 48.32 | 47.95 | 45.39 | 43.12 | 42.81 | 40.58 | 40.46 | 39.65 | 36.12 |
| BiC     |      |      |      |       |       |      |       |      |       |      |       |       |       |       |       |       |       |       |       |       |