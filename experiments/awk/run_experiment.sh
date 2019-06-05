#/bin/bash

#FILE         run_experiment.sh
#AUTHOR       Pavel Prochazka pavel.prochazka@firma.seznam.cz

#Copyright (c) 2019 Seznam.cz, a.s.
#All rights reserved.


## Basic configuration ##
DESCRIPTION="basic experiment - linear position"
DATE=`date '+%Y-%m-%d_%H-%M-%S'`
EXPERIMENT_DIR=/path/to/results/$DATE
VOWPAL_PATH=/path/to/vowpalwabbit/binary
DATASET_PATH=/path/to/the/dataset

MODEL_PATH=$EXPERIMENT_DIR/models
RESULTS_PATH=$EXPERIMENT_DIR/results
LOG_PATH=$EXPERIMENT_DIR/log
LOG_FILE=$LOG_PATH/run_log
SCRIPT_PATH=$EXPERIMENT_DIR/scripts
TEST_DATA=$LOG_PATH/test_data # /dev/null
TRAIN_DATA=$LOG_PATH/train_data # /dev/null


## Initialization ##
mkdir -p $EXPERIMENT_DIR
mkdir -p $LOG_PATH
mkdir -p $MODEL_PATH
mkdir -p $RESULTS_PATH
mkdir -p $SCRIPT_PATH
echo `date '+%Y-%m-%d_%H-%M-%S'` " experiment directory $EXPERIMENT_DIR" | tee >(cat) > $LOG_FILE
cp expand_position.awk vowpal_line_cb.awk vowpal_line_cb_test.awk run_experiment.sh eval_SNIPS.awk gen_pos_metrics.awk $SCRIPT_PATH
echo `date '+%Y-%m-%d_%H-%M-%S'` " backup scripts to $SCRIPT_PATH" | tee >(cat) > $LOG_FILE


## Training Models ##
echo `date '+%Y-%m-%d_%H-%M-%S'` " Training models" | tee >(cat) >> $LOG_FILE
cat $DATASET_PATH/201808??  | ./expand_position.awk | ./vowpal_line_cb.awk |
      tee >($VOWPAL_PATH -f $MODEL_PATH/ips.model --cb 21 --cb_type ips --quiet) \
          >($VOWPAL_PATH -f $MODEL_PATH/dr.model --cb 21 --cb_type dr --quiet) \
          >($VOWPAL_PATH -f $MODEL_PATH/dm.model --cb 21 --cb_type dm --quiet)  > $TRAIN_DATA
echo `date '+%Y-%m-%d_%H:%M:%S'` " Models trained" | tee >(cat) >> $LOG_FILE
sleep 100


## Calculating Predictions ##
echo `date '+%Y-%m-%d_%H-%M-%S'` " Calculating predictions" | tee >(cat) >> $LOG_FILE
cat $DATASET_PATH/201809??  | ./expand_position.awk |
      tee >(./gen_pos_metrics.awk > $RESULTS_PATH/pos_metrics) | ./vowpal_line_cb_test.awk |
            tee >($VOWPAL_PATH -t -i $MODEL_PATH/dr.model -p $RESULTS_PATH/dr_plain.predict --quiet) \
                >($VOWPAL_PATH -t -i $MODEL_PATH/dm.model -p $RESULTS_PATH/dm_plain.predict --quiet) \
                >($VOWPAL_PATH -t -i $MODEL_PATH/ips.model -p $RESULTS_PATH/ips_plain.predict --quiet) > $TEST_DATA
echo `date '+%Y-%m-%d_%H-%M-%S'` " Predictions ready in $RESULTS_PATH" | tee >(cat) >> $LOG_FILE


## Calculation Target Metrics ##
echo `date '+%Y-%m-%d_%H-%M-%S'` " Calculating SNIPS estimates" | tee >(cat) >> $LOG_FILE
paste -d$'\t' $RESULTS_PATH/pos_metrics `ls $RESULTS_PATH/*.predict` | ./eval_SNIPS.awk > $EXPERIMENT_DIR/SNIPS_estimates
i=0
for n in `ls $RESULTS_PATH/*.predict`
do
    echo "method $i: $(basename $n .predict)" >> $EXPERIMENT_DIR/SNIPS_estimates
    i=$(($i+1))
done
echo "model $(($i)): random" >> $EXPERIMENT_DIR/SNIPS_estimates
echo "model $(($i+1)): logging" >> $EXPERIMENT_DIR/SNIPS_estimates
echo `date '+%Y-%m-%d_%H-%M-%S'` " Experiment completed" | tee >(cat) >> $LOG_FILE
echo `date '+%Y-%m-%d_%H-%M-%S'` " Results available in $EXPERIMENT_DIR/SNIPS_estimates" | tee >(cat) >> $LOG_FILE
