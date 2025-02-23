# Comparing online Hyper-parameter tuning variants.
Benchmarking performance of various hyper-parameter tuning algorithms

Assume Rewards are user click reciprocal ranks (continous real reward).

## Algorithm Variants

* [**Thompson Sampler**](https://www.cs.ubc.ca/labs/lci/mlrg/slides/2019_summer_6_thompson_sampling.pdf) with Guassian posterior and Normal-Inverse Gamma priors.
* [**Random Asynchronous Successive Halving Algorithm**](https://arxiv.org/pdf/1810.05934)
* **Factorized Thompson Sampling** which assumes hyper-params are low co-variance, use GP if interactions are expected.
* [**Top-Two Thompson Sampling**](https://arxiv.org/pdf/1602.08448)

## Criteria for Testing
* Sample efficiency
* Latency
* Ease of implementation

## Test
```bash
$ python test.py
```

## Results Preview
```shell
Number of arms: 18750
Real known best arm: 8539
Best combination: [30.   0.3  4.   1.   2.   8. ]

Testing RandomAsynchronousSuccessiveHalvingAlgorithm
#samples: 100, Predicted Best arm: 16141, Rank of predicted best arm: 9503
#samples: 1000, Predicted Best arm: 15745, Rank of predicted best arm: 3341
#samples: 5000, Predicted Best arm: 2058, Rank of predicted best arm: 503
#samples: 10000, Predicted Best arm: 2663, Rank of predicted best arm: 7931
#samples: 50000, Predicted Best arm: 16823, Rank of predicted best arm: 310
#samples: 100000, Predicted Best arm: 13501, Rank of predicted best arm: 276
#samples: 500000, Predicted Best arm: 3314, Rank of predicted best arm: 6

Testing FactorizedThompsonSampler
#samples: 100, Predicted Best arm: 18665, Rank of predicted best arm: 1896
#samples: 1000, Predicted Best arm: 12051, Rank of predicted best arm: 586
#samples: 5000, Predicted Best arm: 1071, Rank of predicted best arm: 398
#samples: 10000, Predicted Best arm: 12488, Rank of predicted best arm: 921
#samples: 50000, Predicted Best arm: 6907, Rank of predicted best arm: 1009
#samples: 100000, Predicted Best arm: 1959, Rank of predicted best arm: 336
#samples: 500000, Predicted Best arm: 10033, Rank of predicted best arm: 804

Testing NormalInverseGammaThompsonSampler
#samples: 100, Predicted Best arm: 6191, Rank of predicted best arm: 663
#samples: 1000, Predicted Best arm: 6669, Rank of predicted best arm: 690
#samples: 5000, Predicted Best arm: 840, Rank of predicted best arm: 5562
#samples: 10000, Predicted Best arm: 4455, Rank of predicted best arm: 987
#samples: 50000, Predicted Best arm: 9064, Rank of predicted best arm: 13
#samples: 100000, Predicted Best arm: 14698, Rank of predicted best arm: 320
#samples: 500000, Predicted Best arm: 13648, Rank of predicted best arm: 94
```