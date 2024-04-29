# Graph-based Semi-Supervised Learning Algorithms

This repository contains implementations of graph-based semi-supervised learning algorithms in Python.

GSSL.py is organized as follows:

## `rgcli` Function

The `rgcli` function performs the Robust Graph that Considers Labeled Instances (RGCLI) algorithm. It constructs a graph based on the input data and performs consistency-based labeling. The method is based on the paper:

- **Title:** RGCLI: Robust Graph that Considers Labeled Instances for Semi-Supervised Learning
- **Authors:** Lilian Berton, Thiago de Paulo Faleiros, Alan Valejo, Jorge Valverde-Rebaza, and Alneu de Andrade Lopes
- **Published in:** Neurocomputing, Volume 226, Pages 238-248, 2017.
- **Available at:** [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231216314680)
- **DOI:** [10.1016/j.neucom.2016.11.053](https://doi.org/10.1016/j.neucom.2016.11.053)

## `llgcl` Function

The `llgcl` function performs label propagation using the Learning with Local and Global Consistency (LLGC) algorithm. It propagates labels through the graph constructed by the RGCLI algorithm. The method is based on the paper:

- **Title:** Learning with local and global consistency
- **Authors:** Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Schölkopf
- **Published in:** Advances in Neural Information Processing Systems, Volume 16, 2003.

## `llgcl_dataset_order` Function

The `llgcl_dataset_order` function orders the dataset separating labeled and unlabeled instances. According to the "Learning with Local and Global Consistency" paper, the first "l" instances correspond to labeled points, where x_i for i<l, with l being the number of labeled instances.

## `GSSLTransductive` Class

The `GSSLTransductive` class implements a Graph-based Semi-Supervised Learning Algorithm in transductive mode.

## `GSSLInductive` Class

The `GSSLInductive` class implements a Graph-based Semi-Supervised Learning Algorithm in inductive mode.

