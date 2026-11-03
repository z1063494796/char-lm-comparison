| Model | Params | Train loss | Val loss | Val perplexity | Gap (val - train) | Train time (s) | Device |
|---|---|---|---|---|---|---|---|
| bigram | 8,281 | 2.4229 | 2.3783 | 10.79 | -0.0446 | 16.7 | Tesla T4 |
| lstm | 956,635 | 1.0715 | 1.1542 | 3.17 | 0.0827 | 118.1 | Tesla T4 |
| transformer | 831,579 | 1.1978 | 1.1935 | 3.3 | -0.0043 | 152.9 | Tesla T4 |
