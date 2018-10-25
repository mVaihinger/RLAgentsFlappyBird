#!/bin/bash

# run evaluation for every experiment setup. For every setup, there are 20x best model and 20x final model:
# logs/experiments/Exp_TYPE/SETUP/
#                                 20 x mthd_outputX
#                                                   best_model
#                                                   final_model
# Evaluate every model on all 20 test environments. Store result of a model in all test environments
# in a csv file "model_results.csv"

# Evaluate converged solution on all 20 test environments of the experiment.
# Store the performance of the final model in the corresponding directory as final_test.csv. This csv file contains
# the performance of the model in all 20 test environments.

# Duration of testing: 0.5M

USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de
NUM_AGENTS=1

#RESULTFILE='test_results.csv'
#TOTAL_TESTSTEPS=500 # 000
#JOB_RESOURCES=nodes=1:ppn=1,pmem=2gb,walltime=00:30:00
JOB_RESOURCES=nodes=1:ppn=1,pmem=2gb,walltime=10:00:00
TOTAL_TESTSTEPS=500000
RESULTFILE='lala.csv'

NENVS=1
#SEED=42
LOG_INTERVAL=0

# Sync data
chmod +x sync_code.sh
./sync_code.sh -u $USERHPC -h $HOST

NAMENL=""
NAMENS=""
NAMENRF=""

# ------- STATIONARY POLICY in OTHER TASKS --------------------------
#TODO how to determine the name of the testing_results.csv file. It should add the setup name: gfNS_test_results
# add flag for name to argparsing and learning function. default value = None. In learning function: if None do the same as before.
EXP_NAME="stationary"

for nl in 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5; do # 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5; do
    NAMENL="-nl$nl"
    RESULTFILE="NL${nl}_test_results.csv"
    echo RESULTFILE

    for rf in 0 ; do # 1 2 3 4; do   # number of random features
        NAMENRF="-nrf$rf"
        TEST_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-test-v0"
        echo $TEST_ENV

#for ns in gfNS gsNS hNS; do
#    NAMENS="-$ns"
#    RESULTFILE="${ns}_test_results.csv"
#    echo RESULTFILE
#    for rf in 0; do   # number of random features
#        NAMENRF="-nrf$rf"
#        TEST_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-test-v0"
#        echo $TEST_ENV
#
        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do #RNDM;
            DIR_NAME="${MTHD}_NRF0"
            echo $EXP_NAME $DIR_NAME

            ARGS=($TEST_ENV $TOTAL_TESTSTEPS $LOG_INTERVAL $RESULTFILE)
            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Testing.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

        done
    done
done

#----------------------------------------------------------
#----------------------------------------------------------


### ---- NOISE -----
#for nl in 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5; do # 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5; do
#    NAMENL="-nl$nl"
#    EXP_NAME="noise"
#    for rf in 0 ; do # 1 2 3 4; do   # number of random features
#        NAMENRF="-nrf$rf"
#        TEST_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-test-v0"
#        echo $TEST_ENV
#        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do # A2C LSTM_A2C GRU_A2C RNDM; do  # TODO
#            DIR_NAME="${MTHD}_NL$nl"

## ---- STATIONARY ---------
#for stat in stat; do
#    EXP_NAME=stationary
#    for rf in 0 2 3 4; do #  ; do   # number of random features
#        NAMENRF="-nrf$rf"
#
#        TEST_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-test-v0"
#        echo $TEST_ENV
#
#        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do
#            DIR_NAME="${MTHD}_NRF$rf"

## ---- NON-STATIONARY ---------
#for ns in gfNS gsNS hNS; do #  gsNS bfNS hNS; do
#    NAMENS="-$ns"
#    EXP_NAME=$ns
#    for rf in 0; do # 1 2 3 4; do   # number of random features
#        NAMENRF="-nrf$rf"
#        TEST_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-test-v0"
#        echo $TEST_ENV
#
#        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do # A2C LSTM_A2C GRU_A2C RNDM; do  # TODO
#            DIR_NAME="${MTHD}_NRF$rf"


#            echo $EXP_NAME $DIR_NAME
#
#            ARGS=($TEST_ENV $TOTAL_TESTSTEPS $LOG_INTERVAL) # $SEED)
#            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
#            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Tracking.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"
#
#        done
#    done
#done