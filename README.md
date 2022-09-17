## Federated K-Means Clustering Algorithm

**A Python project for comparing the proposed greedy centroid initialization with other initialization methods of federated K-means, which includes a centralized/traditional K-means (CKM) and four federated K-means (FKM).**

## Table of Contents

* [Environment requirement](#Environment)
* [Installation](#Installation)
* [Usage](#Usage)
* [Project structure](#Project)
* [Dataset](#Dataset)
* [Contact](#contact)

<!-- * [License](#license) -->

## Environment requirement <a name="Environment"></a>

- Conda 4.10.3 # Conda -V
- Python 3.9.7 # Python3 -V
- Pip3 21.2.4 # Pip3 -V

## Installation  <a name="Installation"></a>
  `$pip3 install -r requirements.txt`

## Project structure <a name="Project"></a>

- docs
- fkm
  - datasets
  - cluster
  - utils
  - vis
  - out
- requirement.txt
- README.md
- UPDATE.md

## Dataset:

- FEMINIST: _downloaded from 'LEAF: https://leaf.cmu.edu/'_
- GAUSSIAN3: _simulated 3 clusters from 3 Gaussian distributions._
- GAUSSIAN10: _simulated 10 clusters from 10 Gaussian distributions._
- MNIST: _handwritten datasets: https://yann.lecun.com/exdb/mnist/ or https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/index.html
- NBAIOT: _IoT dataset: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

## Usage

```shell
$cd fkm/fkm/
$PYTHONPATH='..' python3 main_all.py
```

## Update

- All the update details can be seen in UPDATE.md

## Contact

- Email: kun.bj@outlook.com

[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
