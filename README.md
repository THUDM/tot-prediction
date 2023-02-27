# tot-prediction

## Prerequisites
- Linux
- Python 3.9
- PyTorch 1.10.0+cu111


## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/tot-prediction.git
cd tot-prediction
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

## Dataset
The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1ixgOizcJCBwNF_wNNQsvkQ?pwd=f62u) with password f62u.
Please put the _data_ folder into the project directory.

## How to Run
```bash
python process.py

python citation_only.py  # Use citation number only for prediction
python regressor.py  # Random Forest (RF) and GBRT
python pagerank.py  # PageRank
python gnn.py  # GraphSAGE
```

## Results
Evaluation metrics: average MAP

|       | MAP   |
|-------|-------|
| Citation  | 0.6413 |
| RF | 0.5409 |
| GBRT  | 0.5725 |
| PageRank      |  0.6504     |
| GraphSAGE      |   0.0811     |
