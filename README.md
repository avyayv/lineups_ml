# lineups_ml

Are RAPM models sufficient?

All code is found in `SportsAnalyticsFinalPaper.ipynb`. Train and test data are also found in their respective csv's.

This repository contains code for applying the Transformer architecture to lineup data in the NBA to better understand how players work together. The Transformer, initially designed for sequential data like text, has shown versatility in various domains, including image processing and natural language processing. In this experiment, we adapt the Transformer for lineup analysis, focusing on the interactions between offensive and defensive players.

## Experiment Details

### Tokenization & Positional Encoding
We tokenize the lineup by representing each player as an n-dimensional vector.
To address the order invariance issue, we introduce positional encoding vectors for offensive and defensive players separately.

#### Modified positional encoding: 
- <img src="https://quicklatex.com/cache3/ee/ql_6979922668c610961bd277c582a5daee_l3.png" />
- <img src="https://quicklatex.com/cache3/d3/ql_2443dc33fbd02bda60a0b858c21f03d3_l3.png" /> 
The positional embeddings are added to the original player vectors.

#### Learnable `[CLS]` Embedding
Inspired by BERT, we introduce a learnable `[CLS]` embedding representing the lineup as a whole.
The Transformer Encoder block, with its multi-head attention mechanism, captures team chemistry and interactions between players.

The standard Transformer Encoder Layer is utilized, providing team chemistry through multi-head attention.
The output of each encoder block informs the `[CLS]` embedding, accumulating information about all inputs over multiple blocks.

## Results and Analysis

I compared our approach to a ridge regression approach, and determined that the simpler ridge regression actually performs better in this specific task, as well as being more interepretable. I believe that using a slightly different output variable (i.e expected points per possession, or a distribution of possession outcomes such as steal, shot in paint, etc.) may yield better results. 




