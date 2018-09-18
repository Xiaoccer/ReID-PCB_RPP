# ReID-PCB_RPP
This repository tries to implement the paper:[Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349)
and I almost follow the training details at the origin paper.

## Market-1501 Results
* The result of PCB is little higher than paper.
* The result of PCB+RPP is little lower than paper.
* I think if we do more augmentation with the training data, the performance will be better.

|               |  Feature      | mAP(%)     | Rank-1(%)  |
| ------------- |:-------------:|:----------:| -----:|
| PCB(paper)     |G|77.4|92.3
| PCB(paper)     |H|77.3|92.4
| PCB(ours)      |G|**78.6**|**92.7**
| PCB(ours)      |H|78.5|92.1
| PCB+RPP(paper) |G|81.6|93.8
| PCB+RPP(paper) |H|81.0|93.1
| PCB+RPP(ours)  |G|**80.7**|**92.8**
| PCB+RPP(ours)  |H|79.8|92.4

## Prerequisites
* Python 2.7 or 3.5
* Pytorch 0.4
* GPU Memory

## Getting started
### Dataset
* Dwonload [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html) and extract the files to the current folder
* Change the *download_path* in the `prepare.py` and run `python prepare.py` to prepare the dataset

### Train
Train the model:
```
python train.py --gpu_ids 0,1 --batchsize 64 --data_dir your_data_path --save_dir your_model_save_path --RPP True
```
* `--gpu_ids`: single GPU or mutil-GPUs
* `--batchsize`: batch size
* `--data_dir`: the path of the preparing data
* `--save_dir`: the path to save the training model
* `--RPP`: whether to train with RPP
I train the model on the two GPUs with 64 batchsize
### Test
Test the model(extract the features):
```
python test.py --gpu_ids 0 --which_epoch select the model --stage PCB or full --RPP True or False --feature_H True or False
```
* `--gpu_ids`: just single GPU
* `--which_epoch`: select the i-th model
* `--stage`: select the training model : PCB or full(PCB+RPP)
* `--RPP`: if you choose the PCB model, RPP to False; if you choose the full model, RPP to True
* `--feature_H`: whether to extract the low-dims features

### Evaluate
Evaluate the model
* Evaluate on GPU:
```
python evaluate_gpu.py --gpu_ids 0 --reslut_mat the path of  the features mat
```
* Evaluate on CPU:
```
python evaluate_gpu.py --reslut_mat the path of  the features mat
```

## Reference resources
Thanks to the [layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch/blob/b9465f16a3b94300ceee8884120226804beee224/README.md).
This repository only implements the part of PCB, I made some modifications on it and then add the RPP part.

