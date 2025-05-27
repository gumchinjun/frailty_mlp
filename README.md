# KIST Frailty Classification Project (2024)
The purpose of this project is to classify \[Frail/Prefrail/Robust\] based on EHR data.

![image](https://github.com/user-attachments/assets/c75d3ba5-ebee-4088-8c62-182bb0475f94)


## Getting Started

### Prerequisites
To install packages
```commandline
pip install -r ./requirements.txt
```

### To run the code

#### For 3 class classification

* Robust vs. Prefrail vs. Frail 
```
python run.py --reg_loss_weight 0.8 0.9 0.9
```

#### For 2 class classification

1. \[Robust & Prefrail\] vs. \[Frail\]
```
python run.py --cls_type=2 --two_class_split_type=1 --reg_loss_weight 0.9 0.9 0.8 --l2=1e-4 --scheduler=step
```

2. \[Robust\] vs. \[Prefrail & Frail\]
```
python run.py --cls_type=2 --two_class_split_type=2 --reg_loss_weight 0.8 0.9 0.9 --lr=5e-4
```

3. Robust vs. Frail
```
python run.py --cls_type=2 --two_class_split_type=3 --reg_loss_weight 0.8 0.9 0.9 --l2=1e-3
```

## Results
When the model is properly reproduced, the model should output the following results for the main classification task.
|    Task type    |Acc. (%)|w-F1 (%) |
| --------------- | ------ | ------- |
|     3 class     | 69.76  | 68.94   |
| 2 class, Opt. 1 | 93.13  | 91.99   |
| 2 class, Opt. 2 | 75.95  | 75.95   |
| 2 class, Opt. 3 | 94.81  | 94.81   |
