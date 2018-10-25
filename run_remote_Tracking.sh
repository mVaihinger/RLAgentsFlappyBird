#!/bin/bash
# Load pre trained model and continuously update the model. Higher stepsize, to not converge, but track the
# optimal solution locally.

# Duration of tracking: 0.5M

# TODO Do not store tracking files in different location but name files differently. Therefore remove chages to

USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de
NUM_AGENTS=1

#TOTAL_TESTSTEPS=500 # 000
#JOB_RESOURCES=nodes=1:ppn=1,pmem=2gb,walltime=00:30:00
JOB_RESOURCES=nodes=1:ppn=1,pmem=2gb,walltime=20:00:00
TOTAL_TESTSTEPS=500000

NENVS=1
#SEED=42
LOG_INTERVAL=0
LR=0.005

# Sync data
chmod +x sync_code.sh
./sync_code.sh -u $USERHPC -h $HOST

#NAMENL=""
#NAMENS=""
#NAMENRF=""
#
### ---- NOISE -----
#for nl in 0.2; do # 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5
#    NAMENL="-nl$nl"
#    EXP_NAME="noise"
#    for rf in 0 ; do # 1 2 3 4; do   # number of random features
#        NAMENRF="-nrf$rf"
#        TEST_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-test-v0"
#        echo $TEST_ENV
#        for MTHD in PPO LSTM_PPO GRU_PPO; do  # A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN
#            DIR_NAME="${MTHD}_NL$nl"
#
### ---- STATIONARY ---------
##for stat in stat; do
##    EXP_NAME="stationary"
##    for rf in 0 2 3 4; do #  ; do   # number of random features
##        NAMENRF="-nrf$rf"
##
##        TEST_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-test-v0"
##        echo $TEST_ENV
##
##        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do
##            DIR_NAME="${MTHD}_NRF$rf"
#
#
##for ns in gfNS gsNS hNS; do #  gsNS bfNS hNS; do
##    NAMENS="-$ns"
##    EXP_NAME="$ns"
##    for rf in 0; do # 1 2 3 4; do   # number of random features
##        NAMENRF="-nrf$rf"
##        TEST_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-test-v0"
##        echo $TEST_ENV
##
##        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do
##            DIR_NAME="${MTHD}_NRF$rf"
#            echo $EXP_NAME $DIR_NAME
#
#            ARGS=($TEST_ENV $TOTAL_TESTSTEPS $LOG_INTERVAL $LR) # $SEED)
#            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
#            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Tracking.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"
#
#        done
#    done
#done