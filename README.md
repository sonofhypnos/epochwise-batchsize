# Experiments related to epoch-wise critical periods

This notebook notebook is a modified copy of the [Toy models in superpositions example notebook](https://github.com/timaeus-research/devinterp/blob/main/examples/tms.ipynb) from the devinterp repository.
In this notebook I was trying to test [Garret Baker's hypothesis](https://www.lesswrong.com/posts/DgHjJsxgc2pPpTifG/epoch-wise-critical-periods-and-singular-learning-theory) that there is a relationship between batch size, epochs and temperature. 
Main result: If there is a relationship between the batchsize and the temperature then it is definitely not straightforward. In my experiments the differences between batch sizes where rather small. It looks like while the first transitions happen for lower batch sizes, the higher batch sizes catch up quickly and transition more consistently across different runs. For 6 features in the input-layer there is diminishing returns such that higher batch sizes are worse again (I guess because it does less update steps?), while for 20 features in the input there is no diminishing returns (See plots below).

Credits: [Chen et al. (2023)](https://arxiv.org/abs/2310.06301).
