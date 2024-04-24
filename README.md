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
The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1ixgOizcJCBwNF_wNNQsvkQ?pwd=f62u) with password f62u or [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/influence-prediction/paper-tot-prediction/data.zip).
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


# RGTN-NIE

`cd RGTN-NIE`

## Prerequisites
- Python 3.10
- PyTorch 2.1
- dgl 2.1.0+cu118

## Train
modify `save-path` in `train_geni.sh` and `train_two.sh` to save the model.
* run `sh train_geni.sh` for GENI in tot (full batch training)
* run `sh train_two.sh` for RGTN in tot (full batch training)

## Inference
modify `model_path` in `inference.sh` and `inference_two.sh` to load the model.  
modify `output_dir` in `inference.sh` and `inference_two.sh` to save the prediction results.
* run `sh inference.sh` for GENI in tot (full batch inference)
* run `sh inference_two.sh` for RGTN in tot (full batch inference)

## Evaluation

`python pagerank_nie.py`