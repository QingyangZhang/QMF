# Provable Dynamic Fusion for Low-Quality Multimodal Data

This codebase contains the implementation: 

- **[Provable Dynamic Fusion for Low-Quality Multimodal Data](https://icml.cc/virtual/2023/poster/25229) (ICML2023)** 

## Introduction


<p align="center">
<img src="./illustration.png" width="850" height="320">
</p>

## PyTorch Reimplementation


## Enviroment setup

```
pip install -r requirements.txt
```

## Dataset preparation

1. Download [food101](https://www.kaggle.com/datasets/gianmarco96/upmcfood101) and [MVSA_Single](https://www.kaggle.com/datasets/vincemarcs/mvsasingle) and put them in the folder *datasets*
2. Prepare the train/dev/test splits jsonl files. We follow the [MMBT](https://github.com/facebookresearch/mmbt) settings and provide them in corresponding folders.


Baidu Netdisk:
[food101](https://pan.baidu.com/s/1Tj7jRptTt2V6bxfwrvDSQg) \\
[MVSA_Single](https://pan.baidu.com/s/1URVP8AifWuwIFy6v0uAPOA?pwd=18fw) \\
[nyud2](https://pan.baidu.com/s/1214yDgGeOIbSsWly2MLnuA?pwd=xhq3) \\
[sunrgbd](https://pan.baidu.com/s/1hmrnVDdI_irL3hya1mG1eA?pwd=u9r6)

## Glove model for Bow model (optional)
Download [glove.840B.300d.txt](https://www.kaggle.com/datasets/takuok/glove840b300dtxt) and put it in the folder *datasets/glove_embeds*

## Trained Model
We provide the trained models at [Baidu Netdisk](https://pan.baidu.com/s/1fPltY-QP0YDuthbg89D_aA?pwd=8995).

## Instructions on running experiments
Note: Sheels for reference are provided in the folder *shells*

To run our method on benchmark datasets:
- task="MVSA_Single" or "food101"
- task_type="classification"
- model="latefusion"
- name=$task"_"$model"_model_run_df_$i"
```
python train_df.py --batch_sz 16 --gradient_accumulation_steps 40  \
    --savedir ./saved/$task --name $name  --data_path ./datasets/ \
    --task $task --task_type $task_type  --model $model --num_image_embeds 3 \
    --freeze_txt 5 --freeze_img 3   --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed $i --df true --noise 0.0
```

To run tmc:
```
python train_tmc.py --batch_sz 16 --gradient_accumulation_steps 40  \
    --savedir ./saved/$task --name $name  --data_path ./datasets/ \
    --task $task --task_type $task_type  --model $model --num_image_embeds 3 \
    --freeze_txt 5 --freeze_img 3   --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed $i --df true --noise 0.0
```

To run Others:
- task="MVSA_Single" or "food101"
- task_type="classification"
- model="bow" "bert" "img" "concatbert" "concatbow" "mmbt"
- name=$task"_"$model"_model_run_$i"
```
python train.py --batch_sz 16 --gradient_accumulation_steps 40  \
    --savedir ./saved/$task --name $name  --data_path ./datasets/ \
    --task $task --task_type $task_type  --model $model --num_image_embeds 3 \
    --freeze_txt 5 --freeze_img 3   --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed $i --df true --noise 0.0
```




## Cite
```
@article{zhang2023provable,
  title={Provable Dynamic Fusion for Low-Quality Multimodal Data},
  author={Zhang, Qingyang and Wu, Haitao and Zhang, Changqing and Hu, Qinghua and Fu, Huazhu and Zhou, Joey Tianyi and Peng, Xi},
  journal={arXiv preprint arXiv:2306.02050},
  year={2023}
}
```