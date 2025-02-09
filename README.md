# Comparing online Hyper-parameter tuning variants.
Benchmarking performance of various hyper-parameter tuning algorithms

Assume Rewards are user click reciprocal ranks (continous real reward).

## Algorithm Variants

* [**Thompson Sampler**](https://www.cs.ubc.ca/labs/lci/mlrg/slides/2019_summer_6_thompson_sampling.pdf) with Guassian posterior and Normal-Inverse Gamma priors.
* [**Random Asynchronous Successive Halving Algorithm**](https://arxiv.org/pdf/1810.05934)

## Criteria for Testing
* Sample efficiency
* Latency
* Ease of implementation

## Test
```bash
$ python test.py
```