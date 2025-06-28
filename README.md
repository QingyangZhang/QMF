
# Provable Dynamic Fusion for Low-Quality Multimodal Data

[![Visitor Count](https://komarev.com/ghpvc/?username=QingyangZhang&repo=QMF)](https://github.com/QingyangZhang/QMF)

This repository provides the official implementation for the paper "[Provable Dynamic Fusion for Low-Quality Multimodal Data](https://icml.cc/virtual/2023/poster/25229)" presented at ICML 2023 by Qingyang Zhang and Haitao Wu.

## Highlights

* **Theoretical Framework:** This paper introduces a theoretical framework to understand the criterion for robust dynamic multimodal fusion.
* **Novel Method:** A novel dynamic multimodal fusion method, termed **Quality-aware Multimodal Fusion (QMF)**, is proposed, demonstrating provably better generalization ability.

---

## Environment Setup

To set up the environment, run the following command:

```bash
pip install -r requirements.txt
````

-----

## Dataset Preparation

This project uses two types of multimodal datasets: **Text-Image Classification** and **RGBD Scene Recognition**.

### Text-Image Classification

1.  **Download Datasets:**

      * Download [food101](https://www.kaggle.com/datasets/gianmarco96/upmcfood101)
      * Download [MVSA\_Single](https://www.kaggle.com/datasets/vincemarcs/mvsasingle)
      * Place them in the `datasets` folder.
      * *(Baidu Netdisk links for convenience: [food101](https://pan.baidu.com/s/1Tj7jRptTt2V6bxfwrvDSQg?pwd=5jy4) (pwd: 5jy4), [MVSA\_Single](https://pan.baidu.com/s/1URVP8AifWuwIFy6v0uAPOA?pwd=18fw) (pwd: 18fw))*

2.  **Prepare Splits:** The `train`/`dev`/`test` splits (jsonl files) are prepared following the [MMBT](https://github.com/facebookresearch/mmbt) settings and are provided in their corresponding folders.

3.  **Optional: Pre-trained Models for Text Embeddings:**

      * **Glove:** For the Bow model, download [glove.840B.300d.txt](https://www.kaggle.com/datasets/takuok/glove840b300dtxt) and place it in the `datasets/glove_embeds` folder.
      * **BERT:** For the Bert model, download [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) ([Google Drive Link](https://drive.google.com/file/d/1ivh-3aHtoqRMwVN4ZOPvPm59pFP93-hD/view?usp=sharing)) and place it in the root folder `bert-base-uncased/`.

### RGBD Scene Recognition

1.  **Download Datasets:**
      * Download [NYUD2](https://drive.google.com/file/d/1F_BJ9iAJF8atCgSf1xW1NIZJYMssj0-y/view?usp=drive_link)
      * Download [SUNRGBD](https://drive.google.com/file/d/1XzgYNsez-glZIYMt_6jni2mbVf1mWvT9/view?usp=drive_link)
      * Place them in the `datasets` folder.
      * *(Baidu Netdisk links for convenience: [NYUD2](https://pan.baidu.com/s/1214yDgGeOIbSsWly2MLnuA?pwd=xhq3) (pwd: xhq3), [SUNRGBD](https://pan.baidu.com/s/1HiHRwuGdnFPlZ9gvGyOZEg?pwd=pv6m) (pwd: pv6m))*

-----

## Trained Models

We provide the trained models for download. Please ensure you have the necessary tools to access Baidu Netdisk if using those links.

  * **Trained QMF Models:** [Baidu Netdisk](https://pan.baidu.com/s/1fPltY-QP0YDuthbg89D_aA?pwd=8995) (pwd: 8995)
  * **Pre-trained BERT Model:** [Baidu Netdisk](https://pan.baidu.com/s/1TMg1uiMTZNxKT1O62wgfvg?pwd=zu13) (pwd: zu13)
  * **Pre-trained ResNet18 (for RGB-D tasks):** PyTorch official pre-trained `resnet18` can be downloaded from [this link](https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth).

-----

## Usage Example: Text-Image Classification

Shell scripts for reference are provided in the `shells` folder.

To run our method (**QMF**) on benchmark datasets:

```bash
python train_qmf.py --task "${task}" --noise_level 0.0 --noise_type Gaussian \
```

To evaluate and get the reported accuracy in our paper:

```bash
python train_qmf.py --task "${task}" --epoch 0 --noise_level 5.0 --noise_type Gaussian \
```

To run **TMC** (Trusted Multi-View Classification, ICLR'21):

```bash
# Set parameters
task="MVSA_Single" # or "food101"
task_type="classification"
model="latefusion" # TMC often involves a fusion step, "latefusion" is used as an example base
i=0 # Example seed

name="${task}_tmc_model_run_${i}" # Naming convention for TMC runs

python train_tmc.py --batch_sz 16 --gradient_accumulation_steps 40 \
    --savedir "./saved/${task}" --name "${name}" --data_path "./datasets/" \
    --task "${task}" --task_type "${task_type}" --model "${model}" --num_image_embeds 3 \
    --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed "${i}"
```

To run **Other Baseline Models** (e.g., `bow`, `bert`, `img`, `concatbert`, `concatbow`, `mmbt`):

```bash
# Set parameters
task="MVSA_Single" # or "food101"
task_type="classification"
model="bert" # Choose from: "bow", "bert", "img", "concatbert", "concatbow", "mmbt"
i=0 # Example seed

name="${task}_${model}_model_run_${i}"

python train.py --batch_sz 16 --gradient_accumulation_steps 40 \
    --savedir "./saved/${task}" --name "${name}" --data_path "./datasets/" \
    --task "${task}" --task_type "${task_type}" --model "${model}" --num_image_embeds 3 \
    --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed "${i}" --df true
```

-----

## Citation

If our QMF method or the idea of dynamic multimodal fusion methods are helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{zhang2023provable,
  title={Provable Dynamic Fusion for Low-Quality Multimodal Data},
  author={Zhang, Qingyang and Wu, Haitao and Zhang, Changqing and Hu, Qinghua and Fu, Huazhu and Zhou, Joey Tianyi and Peng, Xi},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

-----

## Acknowledgement

The code implementation is inspired by the following excellent works:

  * [TMC: Trusted Multi-View Classification](https://github.com/hanmenghan/TMC)
  * [Confidence-Aware Learning for Deep Neural Networks](https://github.com/daintlab/confidence-aware-learning)

-----

## Related Works

Here are some interesting works related to this paper:

  * [Uncertainty-based Fusion Netwok for Automatic Skin Lesion Diagnosis](https://ieeexplore.ieee.org/document/9994932/)
  * [Uncertainty Estimation for Multi-view Data: The Power of Seeing the Whole Picture](https://arxiv.org/abs/2210.02676)
  * [Reliable Multimodality Eye Disease Screening via Mixture of Student's t Distributions](https://arxiv.org/abs/2303.09790)
  * [Trusted Multi-Scale Classification Framework for Whole Slide Image](https://arxiv.org/abs/2207.05290)
  * [Fast Road Segmentation via Uncertainty-aware Symmetric Network](https://arxiv.org/abs/2203.04537)
  * [Trustworthy multimodal regression with mixture of normal-inverse gamma distributions](https://arxiv.org/abs/2111.08456)
  * [Uncertainty-Aware Multiview Deep Learning for Internet of Things Applications](https://ieeexplore.ieee.org/document/9906001/)
  * [Automated crystal system identification from electron diffraction patterns using multiview opinion fusion machine learning](https://chemrxiv.org/engage/chemrxiv/article-details/644beb010d87b493e3718ca8)
  * [Trustworthy Long-Tailed Classification](https://arxiv.org/abs/2111.09030)
  * [Trusted multi-view deep learning with opinion aggregation](https://ojs.aaai.org/index.php/AAAI/article/view/20724)
  * [EvidenceCap: Towards trustworthy medical image segmentation via evidential identity cap](https://www.arxiv-vanity.com/papers/2301.00349/)
  * [Federated Uncertainty-Aware Aggregation for Fundus Diabetic Retinopathy Staging](https://arxiv.org/abs/2303.13033)
  * [Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification](https://openaccess.thecvf.com/content/CVPR2022/papers/Han_Multimodal_Dynamics_Dynamical_Fusion_for_Trustworthy_Multimodal_Classification_CVPR_2022_paper.pdf)

-----

## Contact

For any additional questions, feel free to email qingyangzhang@tju.edu.cn.

```
```
