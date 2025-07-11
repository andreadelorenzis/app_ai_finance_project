# Temporal Relational Ranking for Stock Prediction

This project is a Python implementation and analysis of the models presented in the paper **"Temporal Relational Ranking for Stock Prediction"** by Fuli Feng et al. The core objective is to move beyond simple price prediction and instead formulate stock selection as a ranking problem. The models leverage not only the historical time-series data of individual stocks but also the complex relationships between them (e.g., sector/industry co-membership or corporate relationships from Wikidata) to improve prediction accuracy.

## Project Structure

The repository is organized to separate data handling, model definitions, training, and analysis into distinct, manageable components.

.
├── data/
│   ├── 2013-01-01/         # Contains EOD feature data for each stock
│   ├── relation/           # Contains pre-computed relational tensors
│   └── pretrain/           # Directory for saving/loading trained models and embeddings
├── models/
│   ├── lstm.py             # Baseline LSTM model and handler
│   ├── rank_lstm.py        # Rank-LSTM model and handler
│   └── rel_rank_lstm.py    # ReRa-LSTM model and handler
├── notebooks/
│   ├── 1_data_preparation/ # Notebooks for data processing (not included in this context)
│   ├── 2_training.ipynb    # Main notebook for training all models
│   ├── 3_backtesting.ipynb # Notebook for backtesting and plotting IRR movement
│   └── 4_explainability_analysis.ipynb # Notebook for feature importance analysis
└── README.md

## Notebooks Overview

The project is divided into several Jupyter notebooks, each with a specific purpose:

### Data Preparation
Three separate notebooks are responsible for processing raw data:

- `close_prices_data_preparation.ipynb`: Processes raw price data.
- `sector_relation_data.ipynb`: Creates the sector/industry relational tensor.
- `wiki_relation_data.ipynb`: Creates the Wikidata-based relational tensor.

### Model Training (`training.ipynb`)
This is the main notebook for training, evaluation, and analysis. It:

- Implements the three core models: LSTM, Rank-LSTM, and ReRaLSTM.
- Trains the baseline LSTM and Rank-LSTM models.
- Uses the trained Rank-LSTM to generate and save sequential embeddings, a critical input for the final model.
- Trains the final ReRaLSTM model using the generated embeddings and relational data.
- Includes a hyperparameter grid search to find the optimal configuration for Rank-LSTM.

### Experimental Evaluation
Two notebooks are used to analyze the results from different perspectives:

- `backtesting.ipynb`: Loads the pre-trained models and performs backtesting on historical stock data. It visualizes the cumulative Investment Return Ratio (IRR) for various Top-K investment strategies.
- `explainability_analysis.ipynb`: Focuses on model interpretability. It uses correlation analysis and permutation feature importance to identify the most influential features for the Rank-LSTM model.

## Models Implemented

The project demonstrates a progression of model complexity:

### LSTM (Baseline)
A standard Long Short-Term Memory network that processes the time-series features for each stock independently to predict its future return. It is trained with a simple Mean Squared Error (MSE) loss.

### Rank-LSTM
This model enhances the baseline by incorporating a pairwise ranking loss into the training objective. This encourages the model not only to predict accurate return values but also to correctly rank the relative performance of different stocks.

### Relational-Ranking LSTM (ReRa-LSTM)
The most advanced model, which uses pre-computed embeddings from the Rank-LSTM and combines them with explicit relational data. It uses an attention-like mechanism (Temporal Graph Convolution) to weigh the influence of related stocks, aiming to make more informed, context-aware predictions.

## Setup and Installation
### 1. Clone the Repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data and Models
- Ensure the EOD feature data is located in the data/2013-01-01/ directory.
- Ensure the pre-computed relational .npy files are in data/relation/.
- The pre-trained models (.pt files) should be placed in data/pretrain/pretrain/.

## How to Run the Project

The notebooks should be run in the following order to ensure dependencies are met:

1. **Data Preparation Notebooks**  
   If not already done, run the data preparation scripts to generate all necessary feature and relation files.

2. **`training.ipynb`**  
   Run this notebook to train the models. The most crucial output is the sequential embedding file (e.g., `NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy`), which is required by the ReRaLSTM model.

3. **`backtesting.ipynb`**  
   After models are trained, run this notebook to load them and visualize the backtesting performance and summary plots.

4. **`explainability_analysis.ipynb`**  
   Run this notebook to analyze the trained Rank-LSTM model and understand its feature dependencies.

## Key Findings

The results from the notebooks and the paper demonstrate several key insights:

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


| Model | NASDAQ IRR | NYSE IRR |
| :--- | :---: | :---: |
| LSTM | 0.13 | -0.90 |
| Rank-LSTM | 0.68 | 0.56 |
| ReRa-LSTM* | **1.19** | **1.06** |

<p align="center"><i><b>Table 1:</b> Comparison of final IRR for a Top-1 strategy. (*) ReRa-LSTM results use the best-performing relation type for each market (Wikidata for NASDAQ, Sector-Industry for NYSE).</i></p>

<p align="center"><i><b>Plot 1:</b> IRR movement for the three models using a Top-1 strategy. The left plot uses Wikidata relations on NASDAQ, while the right plot uses Sector-Industry relations on NYSE.</i></p>

| Strategy | NASDAQ IRR | NYSE IRR |
| :--- | :---: | :---: |
| Market (Ideal) | 3.40 | 2.42 |
| **ReRa-LSTM (Wiki)** | **1.19** | **0.96** |
| S&P 500 Index | 0.17 | 0.17 |
| DJI Index | 0.22 | 0.22 |

<p align="center"><i><b>Table 2:</b> Comparison of Top-1 strategy IRR for the ReRa-LSTM model against benchmarks.</i></p>

<p align="center"><i><b>Plot 2:</b> IRR movement of ReRa-LSTM for different Top-K investment strategies across the four market-relation scenarios.</i></p>

