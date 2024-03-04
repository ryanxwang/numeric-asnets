# Generalised policy trainer (Numeric Action Schema Networks)

## Installation

The easiest way to install is to follow the instructions under `.devcontainer`,
especially if you are using VS Code. Otherwise installation should mostly just
involve making sure your system has all the dependencies are outlined in
`.devcontainer/Dockerfile`, and then running `post-create.sh`.

## Running a complete experiment

The `experiments_numeric/` subdirectory contains a collection of network/trainer
configurations and sets of PDDL problems to train and test on. For example,
`experiments_numeric/architecture/actprop_2l_comparison_dynamic.py` configures a
three-layer network using comparison modules and dynamic exploration with a 
certain amount of training time, a certain size for hidden units, etc.,
while `experiments/domain/block_grouping.py` describes a subset of block
grouping problems to train on, and a set to test on. You could test the
`actprop_2l_comparison_dynamic` trainer configuration and the `block_grouping`
problem configuration with the following comment:

```
# experiments.actprop_3l is the Python module path for experiments/actprop_3l.py;
# likewise for the second argument.
CUDA_VISIBLE_DEVICES="" ./run_experiment experiments_numeric.architecture.actprop_2l_comparison_dynamic experiments_numeric.domain.block_grouping
```

All results and intermediate working are written to
`experiments-results/P<something>` (the subdirectory name should be reasonably
obvious).

The `collate_results.py` script in `asnets/scripts` can merge the results
produced by `run_experiment.py` into `.json` files that are easy to interpret
for the other tools in `asnets/scripts`.
