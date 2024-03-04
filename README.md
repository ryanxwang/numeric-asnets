# Implementation of Numeric ASNets

This repository contains the code used in the ICAPS'24 paper Learning
Generalised Policies for Numeric Planning. The paper abstract is

> We extend Action Schema Networks (ASNets) to learn generalised policies for
> numeric planning, which features quantitative numeric state variables,
> preconditions and effects. We propose a neural network architecture that can
> reason about the numeric variables both directly and in context of other
> variables. We also develop a dynamic exploration algorithm for more efficient
> training, by better balancing the exploration versus learning tradeoff to
> account for the greater computational demand of numeric teacher planners.
> Experimentally, we find that the learned generalised policies are capable of
> outperforming traditional numeric planners on some domains, and the dynamic
> exploration algorithm to be on average much faster at learning effective
> generalised policies than the original ASNets training algorithm.

This repository is based on the original [ASNets
implementation](https://github.com/qxcv/asnets) and inherits its structure:

- `asnets/` contains our implementation and experiment files. Consult
  [`asnets/README.md`](https://github.com/thyroidr/numeric-asnets/blob/master/asnets/README.md)
  for instructions on installing and running the code.
- `problems/numeric` includes all problems that we used to train + test the
  network, plus some problems which might be helpful for further research or
  debugging.

If you use this code in an academic publication, we'd appreciate it if you cited the following paper:

```bibtex
@inproceedings{wang2024numericasnets,
  title={Learning Generalised Policies for Numeric Planning},
  author={Wang, Ryan Xiao and Thi{\'e}baux, Sylvie},
  booktitle={International Conference on Automated Planning and Scheduling},
  year={2024}
}
```

Comments & queries can go to [Ryan Wang](mailto:ryanxiaowang2001@gmail.com).
