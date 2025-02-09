# Comparing online Hyper-parameter tuning variants.
Benchmarking performance of various hyper-parameter tuning algorithms

Assume Rewards are user click reciprocal ranks (continous real reward).

## Algorithm Variants

* [**Thompson Sampler**](https://www.cs.ubc.ca/labs/lci/mlrg/slides/2019_summer_6_thompson_sampling.pdf) with Guassian posterior and Normal-Inverse Gamma priors.
* [**Random Asynchronous Successive Halving Algorithm**](https://arxiv.org/pdf/1810.05934)
* **Factorized Thompson Sampling** which assumes the hyper-params are indenpendant without interaction, using GP if interactions are expected.

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
Real known best arm: 10902
Best combination: [90.   0.2  2.   1.   0.   2. ]

Testing RandomAsynchronousSuccessiveHalvingAlgorithm
#samples: 100, Predicted Best arm: 9896, Rank of predicted best arm: 9091
#samples: 1000, Predicted Best arm: 574, Rank of predicted best arm: 13827
#samples: 5000, Predicted Best arm: 17491, Rank of predicted best arm: 1635
#samples: 10000, Predicted Best arm: 254, Rank of predicted best arm: 11662
#samples: 50000, Predicted Best arm: 13472, Rank of predicted best arm: 1140
#samples: 100000, Predicted Best arm: 2711, Rank of predicted best arm: 756
#samples: 500000, Predicted Best arm: 15737, Rank of predicted best arm: 89

Testing FactorizedThompsonSampler
#samples: 100, Predicted Best arm: 17733, Rank of predicted best arm: 3101                                                                                                                                              
#samples: 1000, Predicted Best arm: 3842, Rank of predicted best arm: 2361                                                                                                                                              
#samples: 5000, Predicted Best arm: 10386, Rank of predicted best arm: 604                                                                                                                                              
#samples: 10000, Predicted Best arm: 15671, Rank of predicted best arm: 1164                                                                                                                                            
#samples: 50000, Predicted Best arm: 6276, Rank of predicted best arm: 283                                                                                                                                              
#samples: 100000, Predicted Best arm: 2682, Rank of predicted best arm: 2356                                                                                                                                            
#samples: 500000, Predicted Best arm: 9621, Rank of predicted best arm: 242 

Testing NormalInverseGammaThompsonSampler
#samples: 100, Predicted Best arm: 12461, Rank of predicted best arm: 3532                                                                                                                                              
#samples: 1000, Predicted Best arm: 178, Rank of predicted best arm: 41                                                                                                                                                 
#samples: 5000, Predicted Best arm: 8889, Rank of predicted best arm: 1132
```