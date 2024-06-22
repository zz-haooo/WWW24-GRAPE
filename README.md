# WWW 2024 Towards Expansive and Adaptive Hard Negative Mining: Graph Contrastive Learning via Subspace Preserving

Implementation of WWW 24 paper GRAPE(https://dl.acm.org/doi/abs/10.1145/3589334.3645327).

# Environment

The common python library configuration for GNN with dgl is enough to run the code.


Or you can configure a new Python environment with Anaconda as follows:

```shell
conda create -n grape python=3.8
conda activate grape
conda install --yes --file requirements.txt
```


# Running

We have provided a `run.sh` file, and you can execute the commands within it to verify our results.


## Acknowledgements

The code is implemented partially based on [CCA-SSG](https://github.com/hengruizhang98/CCA-SSG).

## Citation

If you find our codes useful, please consider citing our work

```
@inproceedings{hao2024towards,
  title={Towards Expansive and Adaptive Hard Negative Mining: Graph Contrastive Learning via Subspace Preserving},
  author={Hao, Zhezheng and Xin, Haonan and Wei, Long and Tang, Liaoyuan and Wang, Rong and Nie, Feiping},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={322--333},
  year={2024}
}
