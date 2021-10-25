# Meta-SparseINR

Official PyTorch implementation of **"Meta-learning Sparse Implicit Neural Representations"** (NeurIPS 2021) by [Jaeho Lee*](https://jaeho-lee.github.io/), [Jihoon Tack*](https://jihoontack.github.io/), [Namhoon Lee](https://www.robots.ox.ac.uk/~namhoon/), and [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html).

**TL;DR**: *We develop a scalable method to learn sparse neural representations for a large set of signals.*

<p align="center">
    <img src=figures/method_overview.png width="900"> 
</p>

Illustrations of (a) an implicit neural representation, (b) the standard pruning algorithm that prunes and retrains the model for each signal considered, and (c) the proposed Meta-SparseINR procedure to find a sparse initial INR, which can be trained further to fit each signal.

## 1. Requirements
```
conda create -n inrprune python=3.7
conda activate inrprune

conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

pip install torchmeta
pip install imageio einops tensorboardX
```

### Datasets
- Download Imagenette and SDF file from the following page:
    - Imagenette: [fastai](https://github.com/fastai/imagenette/)
    - SDF: [Learnit](https://www.matthewtancik.com/learnit)
- One should locate the dataset into `/data` folder

## 2. Training

### Training option
The option for the training method is as follows:
- `<DATASET>`: {`celeba`,`sdf`,`imagenette`}

### Meta-SparseINR (ours)
```
# Train dense model first
python main.py --exp meta_baseline --epoch 150000 --data <DATASET>

# Iterative pruning (magnitude pruning)
python main.py --exp metaprune --epoch 30000 --pruner MP --amount 0.2 --data <DATASET>
```

### Random Pruning
```
# Train dense model first
python main.py --exp meta_baseline --epoch 150000 --data <DATASET>

# Iterative pruning (random pruning)
python main.py --exp metaprune --epoch 30000 --pruner RP --amount 0.2 --data <DATASET>
```

### Dense-Narrow
```
# Train dense model with a given width

# Shell script style
widthlist="230 206 184 164 148 132 118 106 94 84 76 68 60 54 48 44 38 34 32 28"
for width in $widthlist
do
    python main.py --exp meta_baseline --epoch 150000 --data <DATASET> --width $width --id width_$width
done
```

## 3. Evaluation
### Evaluation option
The option for the training method is as follows:
- `<DATASET>`: {`celeba`,`sdf`,`imagenette`}
- `<OPT_TYPE>`: {`default`,`two_step_sgd`}, default denotes adam optimizer with 100 steps.

We assume all checkpoints are trained.

### Meta-SparseINR (ours)
```
python eval.py --exp prune --pruner MP --data <DATASET> --opt_type <OPT_TYPE>
```

### Baselines
```
# Random pruning
python eval.py --exp prune --pruner RP --data <DATASET> --opt_type <OPT_TYPE>

# Dense-Narrow
python eval.py --exp dense_narrow --data <DATASET> --opt_type <OPT_TYPE>

# MAML + One-Shot
python eval.py --exp one_shot --data <DATASET> --opt_type default

# MAML + IMP
python eval.py --exp imp --data <DATASET> --opt_type default

# Scratch
python eval.py --exp scratch --data <DATASET> --opt_type <OPT_TYPE>
```

## 4. Experimental Results

<p align="center">
    <img src=figures/results.png width="900"> 
</p>

<p align="center">
    <img src=figures/visualcomp.png width="900"> 
</p>

## Citation
```
@inproceedings{lee2021meta,
  title={Meta-learning Sparse Implicit Neural Representations},
  author={Jaeho Lee and Jihoon Tack and Namhoon Lee and Jinwoo Shin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Reference
- [JAX Learnit](https://github.com/tancik/learnit)
- [PyTorch Meta Learning](https://github.com/tristandeleu/pytorch-meta)
- [PyTorch Siren](https://github.com/lucidrains/siren-pytorch)
- [PyTorch Occupancy Network](https://github.com/autonomousvision/occupancy_networks)
- [PyTorch MetaSDF](https://github.com/vsitzmann/metasdf)
