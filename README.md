# Point-MAE

## Masked Autoencoders for Point Cloud Self-supervised Learning, [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620591.pdf), [ArXiv](https://arxiv.org/abs/2203.06604)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/masked-autoencoders-for-point-cloud-self/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=masked-autoencoders-for-point-cloud-self)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/masked-autoencoders-for-point-cloud-self/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=masked-autoencoders-for-point-cloud-self)

In this work, we present a novel scheme of masked autoencoders for point cloud self-supervised learning, termed as Point-MAE. Our Point-MAE is neat and efficient, with minimal modifications based on the properties of the point cloud. In classification tasks, Point-MAE outperforms all the other self-supervised learning methods on ScanObjectNN and ModelNet40. Point-MAE also advances state-of-the-art accuracies by 1.5%-2.3% in the few-shot learning on ModelNet40. 

<div  align="center">    
 <img src="./figure/net.jpg" width = "666"  align=center />
</div>

## 1. Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```

For Linux Kernal 6.0 or above (e.g. Ubuntu 24), please run the following command before installing Chamfer Distance:
```
sudo apt install gcc-10 g++-10

su
cd /usr/local/src
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.tar.xz
tar -xf linux-5.4.tar.xz && cd linux-5.4
make headers_install INSTALL_HDR_PATH=/usr/local/linux-headers-5.4

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CFLAGS="-I/usr/local/linux-headers-5.4/include"
export CPPFLAGS="-I/usr/local/linux-headers-5.4/include"
```

In `extensions/chamfer_dist/setup.py`, in the `extra_compile_args` field, pass the correct header path to nvcc by adding the following line as the second element of `ext_modules`:
```
extra_compile_args={"nvcc": ['--system-include=/usr/local/linux-headers-5.4/include']}
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

## 3. Point-MAE Models
|  Task | Dataset | Config | Acc.| Download|      
|  ----- | ----- |-----|  -----| -----|
|  Pre-training | ShapeNet |[pretrain.yaml](./cfgs/pretrain.yaml)| N.A. | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth) |
|  Classification | ScanObjectNN |[finetune_scan_hardest.yaml](./cfgs/finetune_scan_hardest.yaml)| 85.18%| [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_hardest.pth)  |
|  Classification | ScanObjectNN |[finetune_scan_objbg.yaml](./cfgs/finetune_scan_objbg.yaml)|90.02% | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objbg.pth) |
|  Classification | ScanObjectNN |[finetune_scan_objonly.yaml](./cfgs/finetune_scan_objonly.yaml)| 88.29%| [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objonly.pth) |
|  Classification | ModelNet40(1k) |[finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml)| 93.80%| [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_1k.pth) |
|  Classification | ModelNet40(8k) |[finetune_modelnet_8k.yaml](./cfgs/finetune_modelnet_8k.yaml)| 94.04%| [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_8k.pth) |
| Part segmentation| ShapeNetPart| [segmentation](./segmentation)| 86.1% mIoU| [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/part_seg.pth) |

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

Our codes are built upon [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Reference

```
@inproceedings{pang2022masked,
  title={Masked autoencoders for point cloud self-supervised learning},
  author={Pang, Yatian and Wang, Wenxiao and Tay, Francis EH and Liu, Wei and Tian, Yonghong and Yuan, Li},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part II},
  pages={604--621},
  year={2022},
  organization={Springer}
}
```
