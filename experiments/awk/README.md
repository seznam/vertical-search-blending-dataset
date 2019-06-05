# Scripts to generate the results in the paper

### Requirements
- bash
- awk
- [vowpal wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) – the experiments were conducted with version 8.6.1

### Configuration
The configuration is set at the top of the main script `run_experiment.sh`. In particular:
```
## Basic configuration ##
DESCRIPTION="basic experiment - linear position" # string describing the experiment
DATE=`date '+%Y-%m-%d_%H-%M-%S'` 
EXPERIMENT_DIR=/path/to/results/$DATE # where to save the experiment log and output
VOWPAL_PATH=/path/to/vowpalwabbit/binary
DATASET_PATH=/path/to/the/dataset
```

### Results
To run the experiment, just configure `run_experiment.sh` and run it. The main results are saved to `SNIPS_estimates` file.
