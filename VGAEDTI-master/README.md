# VGAELDA: a representation learning model based on variational inference and graph autoencoder for predicting lncRNA-disease associations

Code for our paper "[A representation learning model based on variational inference and graph autoencoder for predicting lncRNA-disease associations](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04073-z)"

## Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:

```
numpy==1.16.5
pytorch==1.3.1
sklearn==0.21.3
```

## Usage

```bash
git clone https://github.com/zhanglabNKU/VGAELDA.git
cd VGAELDA
python fivefoldcv.py --data 1
```

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Dimension of representations')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight between lncRNA space and disease space')
parser.add_argument('--data', type=int, default=1, choices=[1,2],
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
```

## Datasets

Files in Dataset1 are listed as follow:

- `lncRNA_115.txt`  includes the names of all 115 lncRNAs in Dataset1.
- `disease_178.txt`  includes the names of all 178 diseases in Dataset1.
- `known_lncRNA_disease_interaction.txt` is a 115x178 matrix  `Y`  that shows lncRNA-disease associations. `Y[i,j]=1`  if lncRNA `i`  and disease `j` are known to be associated, otherwise 0.
- `known_gene_disease_interaction.txt` is the feature matrix of diseases.
- `rnafeat.txt` is the feature matrix of lncRNAs.

Files in Dataset2 are defined similarly to Dataset1.

## Citation

```
@article{shi2021vgaelda,
    author={Zhuangwei Shi and Han Zhang and Chen Jin and Xiongwen Quan and Yanbin Yin},
    title={A representation learning model based on variational inference and graph autoencoder for predicting lncRNA-disease associations},
    journal={BMC Bioinformatics},
    year={2021},
    volume={22},
    number={136},
    pages={1-20},
    url={https://doi.org/10.1186/s12859-021-04073-z},
}
```
