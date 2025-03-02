# Comparing online Hyper-parameter tuning variants.
Benchmarking performance of various hyper-parameter tuning algorithms

Assume Rewards are user click reciprocal ranks (continous real reward).

## Algorithm Variants

* [**Thompson Sampler**](https://www.cs.ubc.ca/labs/lci/mlrg/slides/2019_summer_6_thompson_sampling.pdf) with Guassian posterior and Normal-Inverse Gamma priors.
* [**Random Asynchronous Successive Halving Algorithm**](https://arxiv.org/pdf/1810.05934)
* **Factorized Thompson Sampling** which assumes hyper-params are low co-variance, use Guassian Process if interactions are expected.
* [**Top-Two Thompson Sampling**](https://arxiv.org/pdf/1602.08448)
* [**Population Based Search**](https://arxiv.org/pdf/1711.09846)

## Variants / Research for prototyping
- [**CMA-ES with margin**](https://arxiv.org/pdf/2305.00849v1)
  - And Other [variants](https://arxiv.org/search/cs?searchtype=author&query=Shirakawa,+S)
- [**Streaming Sparse Gaussian Process Approximations**](https://proceedings.neurips.cc/paper_files/paper/2017/file/f31b20466ae89669f9741e047487eb37-Paper.pdf)
- [**Limited-Memory Matrix Adaptation
for Large Scale Black-box Optimization**](https://arxiv.org/pdf/1705.06693)
- [**Illuminating search spaces by mapping elites**](https://arxiv.org/pdf/1504.04909)

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
Real known best arm: 15153
Best combination: [120.     0.49   1.     1.     0.     4.  ]

Testing RandomAsynchronousSuccessiveHalvingAlgorithm
#samples: 100, Predicted Best arm: 1589, Rank of predicted best arm: 13837
#samples: 1000, Predicted Best arm: 17296, Rank of predicted best arm: 17469
#samples: 5000, Predicted Best arm: 15233, Rank of predicted best arm: 876
#samples: 10000, Predicted Best arm: 15026, Rank of predicted best arm: 50
#samples: 50000, Predicted Best arm: 15528, Rank of predicted best arm: 274
#samples: 100000, Predicted Best arm: 15060, Rank of predicted best arm: 207
#samples: 500000, Predicted Best arm: 15203, Rank of predicted best arm: 127

Testing FactorizedThompsonSampler
#samples: 100, Predicted Best arm: 16468, Rank of predicted best arm: 17593
#samples: 1000, Predicted Best arm: 697, Rank of predicted best arm: 13663
#samples: 5000, Predicted Best arm: 15128, Rank of predicted best arm: 7
#samples: 10000, Predicted Best arm: 15153, Rank of predicted best arm: 1

Testing NormalInverseGammaThompsonSampler
#samples: 100, Predicted Best arm: 10993, Rank of predicted best arm: 4601
#samples: 1000, Predicted Best arm: 5834, Rank of predicted best arm: 12003
#samples: 5000, Predicted Best arm: 12925, Rank of predicted best arm: 2090
#samples: 10000, Predicted Best arm: 15153, Rank of predicted best arm: 1

Testing TopTwoNormalInverseGammaThompsonSampler
#samples: 100, Predicted Best arm: 6989, Rank of predicted best arm: 7264
#samples: 1000, Predicted Best arm: 5108, Rank of predicted best arm: 11691
#samples: 5000, Predicted Best arm: 11494, Rank of predicted best arm: 5274
#samples: 10000, Predicted Best arm: 14332, Rank of predicted best arm: 1578
#samples: 50000, Predicted Best arm: 15153, Rank of predicted best arm: 1
```