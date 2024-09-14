# LANE: Can Non-tuned Large Language Models Achieve Logic Alignment? An Explainable ReasonGeneration Method for Recommendation System

This is our Pytorch implementation for the paper:

Can Non-tuned Large Language Models Achieve Logic Alignment? An Explainable ReasonGeneration Method for Recommendation System. ([Earlier versions](https://arxiv.org/abs/2407.02833))

Please cite our paper if you use the code.

##  Installation

1. **clone the repositoru**

```
git clone https://github.com/Ming209/LANE.git
cd LANE
```

2. **Requirement**


Use the following command to install the required dependencies:

```
pip install -r requirements.txt
```

3. **Datesets**

You can download the three real-world datasets we used from the link below.

- MovieLens: [ml-1m.zip](https://files.grouplens.org/datasets/movielens/ml-1m.zip)

- Amazon: [reviews_Books.json.gz](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz), [meta_Books.json.gz](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz)

- Steam: [steam_reviews.json.gz](http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz), [steam_games.json.gz](http://cseweb.ucsd.edu/~wckang/steam_games.json.gz)

Before running the code, you need to put the downloaded dataset into the corresponding folder under the `data/dateset/raw` path.

## Usage

For simplicity, here we take ml-1m as an example：

1. **Data Preparation**

Use the following code to process different datasets into a unified txt format. You can find them in the `data/dataset/serialize` folder:

```
python data/script/serialize_Beauty.py
```

Then use the following code to process the txt data to get the preprocessed data.You can find them in the `data/preprocessed` folder:

```
python Datapreprocessed.py --dataset=Beauty
```

>[!NOTE]
>the GPT-3.5-turbo API will be called here, and you need to fill in the **API key** in the *gpt_request()* function in `utils.py` in advance

2. **Train**

To train **Baseline(SASRec)** on Beauty (with default hyper-parameters):

```
python train_baseline.py --dataset=Beauty --model=SASRec --config_path=config/sasrec.json --maxlen=50
```

To train **LANE(LANE-SASRec)** on Beauty (with default hyper-parameters):

```
python train_LANE.py --dataset=Beauty --inte_model=SASRec --inte_model_config_path=config/sasrec.json --maxlen=50
```

# Reference

```
@article{zhao2024lane,
  title={Lane: Logic alignment of non-tuning large language models and online recommendation systems for explainable reason generation},
  author={Zhao, Hongke and Zheng, Songming and Wu, Likang and Yu, Bowen and Wang, Jing},
  journal={arXiv preprint arXiv:2407.02833},
  year={2024}
}
```