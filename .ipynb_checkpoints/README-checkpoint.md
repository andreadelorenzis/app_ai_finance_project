# Temporal Relational Ranking for Stock Prediction

This project is a Python implementation and analysis of the models presented in the paper **"Temporal Relational Ranking for Stock Prediction"** by Fuli Feng et al. The core objective is to move beyond simple price prediction and instead formulate stock selection as a ranking problem. The models leverage not only the historical time-series data of individual stocks but also the complex relationships between them to improve prediction accuracy using Graph Neural Networks. 

## Models Implemented

### LSTM (Baseline)
A standard Long Short-Term Memory network that processes the time-series features for each stock independently to predict its future return. It is trained with a simple Mean Squared Error (MSE) loss.

### Rank-LSTM
This model enhances the baseline by incorporating a pairwise ranking loss into the training objective. This encourages the model not only to predict accurate return values but also to correctly rank the relative performance of different stocks.

### Relational-Ranking LSTM (ReRa-LSTM)
The most advanced model, which uses pre-computed embeddings from the Rank-LSTM and combines them with explicit relational data. It uses an attention-like mechanism (Temporal Graph Convolution) to weigh the influence of related stocks, aiming to make more informed, context-aware predictions.

## Data and Models
All data (Sequential data, Industry and Wiki relation) are under the `data/` folder.

### Sequential Data
**Raw data:** `data/google_finance` 
Historical (30 years) End-of-day data (i.e., open, high, low, close prices and trading volume) of more than 8,000 stocks traded in US stock market collected from Google Finance.

**Processed data:** `data/2013-01-01`
Is the dataset used to conducted the experiments.

### Relation Data
To get the relation data, run the following command in the `data/` folder:
```
tar zxvf relation.tar.gz
```

**Industry Relation**
Under the sector_industry folder, there are row relation file and binary encoding file (.npy) storing the industry relations between stocks in NASDAQ and NYSE.

**Wiki Relation**
Under the wikidata folder, there are row relation file and binary encoding file (.npy) storing the Wiki relations between stocks in NASDAQ and NYSE.

## Project Structure

- `data`
	- `2013-01-01/`: contains End-of-Day pre-processed feature data for each stock
	- `relation/`: contains pre-computed relational tensors to feed into the GNN
	- `pretrain/`: directory for saving/loading trained models and embeddings
- `notebooks`
	- `preprocessing/`: notebooks for data collection and preprocessing
	- `training/`: notebooks to define and train models
	- `experiments/`: notebooks for model evaluation on test data
- `models` 
	- `lstm.py`: baseline LSTM model and trainer
	- `rank_lstm.py`: Rank-LSTM model and trainer
	- `rel_rank_lstm.py`: ReRa-LSTM model and trainer

## Notebooks Overview

### Data Preparation
Three separate notebooks are responsible for processing raw data:

- `01_close_prices_data_preparation.ipynb`: Processes raw price data.
- `02_sector_relation_data`: Creates the sector/industry relational tensor.
- `03_wiki_relation_data`: Creates the Wikidata-based relational tensor.

### Model Training

- `01_data_loading_and_metrics`: implements the data loading and evaluation functions.
- `02_LSTM_model_definition`: LSTM model definition and training
- `03_rank_LSTM_model_definition`: Rank-LSTM model definition and training. Used to generate and save the sequential embeddings for the final TGNN model. 
- `04_rel_rank_LSTM_model_definition`: ReRaLSTM model definition and training, using the generated rank LSTM embeddings and relational data.
- `05_hyperparameter_optimization`: performs a hyperparameter grid search to find the optimal configuration for the models.

### Experimental Evaluation

- `backtesting.ipynb`: loads the pre-trained models and performs backtesting on historical stock data. It visualizes the cumulative Investment Return Ratio (IRR), comparing the performance of the models and of various investment strategies.
- `explainability_analysis.ipynb`: model interpretability. It uses correlation analysis and permutation feature importance to identify the most influential features for the Rank-LSTM model.

## Setup and Installation
### 1. Clone the Repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run the Project
To repeat the experiment (train a RSR model), download the pretrained [sequential embedding](https://drive.google.com/file/d/1fyNCZ62pEItTQYEBzLwsZ9ehX_-Ai3qT/view?usp=sharing), and extract the file into the data folder.

The notebooks should be run in the following order to ensure dependencies are met:

1. **Data Preparation Notebooks**  
   If not already done, run the data preparation scripts to generate all necessary feature and relation files.

2. **Models Training Notebooks**
   Run this notebooks to train the models. The most crucial output is the sequential embedding file (e.g., `NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy`), which is required by the ReRaLSTM model (if not already downloaded like mentioned above).

3. **Experimental Evaluation Notebooks**  
   After models are trained, run this notebook to load them and visualize the backtesting performance and summary plots.

## Key Findings

The results from the notebooks demonstrate several key insights:

- **Ranking is Superior**  
  Formulating the problem with a ranking loss (Rank-LSTM) consistently yields a higher Investment Return Ratio (IRR) than a simple regression-based LSTM.

- **Relations Add Value**  
  Incorporating relational data via the ReRa-LSTM model further improves performance, especially in the more stable NYSE market.

- **Quality of Relations Matters**  
  The performance lift is highly dependent on the type of relational data.  
  - For the volatile NASDAQ market, the complex Wikidata relations provided a significant boost.  
  - The simpler Sector-Industry relations were less effective.  
  - Conversely, the more stable NYSE market benefited most from the long-term industry correlations.

- **Strategy Affects Performance**  
  Backtesting shows that a Top-1 strategy (investing in the single best-predicted stock) often outperforms Top-5 and Top-10 strategies, indicating that the model is effective at identifying the highest-potential assets.

### Model Performance Comparison
| Model           | NASDAQ MSE | NASDAQ MRR | NASDAQ IRR | NYSE MSE | NYSE MRR | NYSE IRR |
|-----------------|------------|------------|------------|----------|----------|----------|
| LSTM            | 0.0004     | 6.07e-03   | 0.80       | 0.0004   | 9.85e-03 | 1.27     |
| Rank-LSTM       | 0.3713     | 2.13e-02   | 1.55       | 0.3673   | 5.96e-03 | 1.42     |
| ReRa-LSTM       | 0.0004     | 3.39e-02   | 2.22       | 0.0002   | 3.32e-02 | 1.62     |

### Investment Strategy Comparison
| Strategy  | NASDAQ Top-1 | NASDAQ Top-5 | NASDAQ Top-10 | NYSE Top-1 | NYSE Top-5 | NYSE Top-10 |
| --------- | ------------ | ------------ | ------------- | ---------- | ---------- | ----------- |
| Market    | 40.89        | 24.53        | 19.05         | 35.35      | 21.74      | 17.12       |
| ReRa-LSTM | 1.22         | 0.30         | 0.09          | 0.62       | 0.35       | 0.23        |
| S&P 500   | 0.16         | 0.16         | 0.16          | 0.16       | 0.16       | 0.16        |
| DJI       | 0.20         | 0.20         | 0.20          | 0.20       | 0.20       | 0.20        |
