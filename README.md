# Point-MAE

## Implementation for paper: Masked Autoencoders for Point Cloud Self-supervised Learning

In this work, we present a novel scheme of masked autoencoders for point cloud self-supervised learning, termed as Point-MAE. Our Point-MAE is neat and efficient, with minimal modifications based on the properties of the point cloud. In classification tasks, Point-MAE outperforms all the other self-supervised learning methods on ScanObjectNN and ModelNet40. Point-MAE also advances state-of-the-art accuracies by 1.5%-2.3% in the few-shot learning on ModelNet40. 

<div  align="center">    
 <img src="./figure/net.jpg" width = "666"  align=center />
</div>

## 1. Requirements
PyTorch >= 1.7.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

## 3. Point-MAE Models
|  Task | Dataset | Config | Acc.| Download|      
|  ----- | ----- |-----|  -----| -----|
|  Pre-training | ShapeNet |[pretrain.yaml](./cfgs/pretrain.yaml)| N.A. | |
|  Classification | ScanObjectNN |[finetune_scan_hardest.yaml](./cfgs/finetune_scan_hardest.yaml)| 84.52%| |
|  Classification | ScanObjectNN |[finetune_scan_objbg.yaml](./cfgs/finetune_scan_objbg.yaml)| 88.29%| |
|  Classification | ScanObjectNN |[finetune_scan_objonly.yaml](./cfgs/finetune_scan_objonly.yaml)| 90.01%| |
|  Classification | ModelNet40(1k) |[finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml)| 93.80%| |
|  Classification | ModelNet40(8k) |[finetune_modelnet_8k.yaml](./cfgs/finetune_modelnet_8k.yaml)| 94.04%| |
| Part segmentation| ShapeNetPart| [segmentation](./segmentation)| 86.1% mIoU| |

|  Task | Dataset | Config | 5w10s Acc. (%)| 5w20s Acc. (%)| 10w10s Acc. (%)| 10w20s Acc. (%)|     
|  ----- | ----- |-----|  -----| -----|-----|-----|
|  Few-shot learning | ModelNet40 |[fewshot.yaml](./cfgs/fewshot.yaml)| 96.3 ± 2.5| 97.8 ± 1.8| 92.6 ± 4.1| 95.0 ± 3.0| 

## 4. Point-MAE Pre-training
To pretrain Point-MAE on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain.yaml --exp_name <output_file_name>
```
## 5. Point-MAE Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Voting on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```

## 6. Visualization

Visulization of pre-trained model on ShapeNet validation set, run:

```
python main_vis.py --test --ckpts <path/to/pre-trained/model> --config cfgs/pretrain.yaml --exp_name <name>
```

<div  align="center">    
 <img src="./figure/vvv.jpg" width = "900"  align=center />
</div>

## Acknowledgements

Our codes are build upon [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Reference
