#!/bin/bash
USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de

NENVS=1
MAX_GRAD_NORM=0.01

#TOTAL_TRAINSTEPS=2000
#NUM_AGENTS=3 # remove comment from run_Training.sh
#JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=00:30:00
#LOG_INTERVAL=10
#TEST_INTERVAL=10  # TODO

#TOTAL_TRAINSTEPS=2000000
TOTAL_TRAINSTEPS=500000
NUM_AGENTS=20 # remove comment from run_Training.sh
JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=24:00:00
LOG_INTERVAL=0  #3000
TEST_INTERVAL=300

# Sync data
chmod +x sync_code.sh
./sync_code.sh -u $USERHPC -h $HOST

NAMENL=""
NAMENS=""
NAMENRF=""

## ---- NOISE -----
#for nl in 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5; do # ; do
#    NAMENL="-nl$nl"
#    EXP_NAME="noise"
#    for rf in 0; do #  2 3 4; do   # number of random features
#        NAMENRF="-nrf$rf"
#        TRAIN_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-train-v0"
#        echo $TRAIN_ENV
#        TEST_ENV="ContFlappyBird-clip$NAMENL$NAMENS$NAMENRF-test-v0"
#        echo $TEST_ENV
#        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do # A2C LSTM_A2C GRU_A2C RNDM; do  # TODO
#            DIR_NAME="${MTHD}_NL$nl"

# ---- STATIONARY ---------
for stat in stat; do
    EXP_NAME=stationary
    for rf in 0 2 3 4; do   # number of random features
        NAMENRF="-nrf$rf"

#        TRAIN_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-train-v0"
#        echo $TRAIN_ENV
#        TEST_ENV="ContFlappyBird-clip$NAMENL$NAMENS$NAMENRF-test-v0"
#        echo $TEST_ENV
#
#        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do
#            DIR_NAME="${MTHD}_NRF$rf"
#
## ---- NON-STATIONARY --------
#for ns in gfNS gsNS hNS; do #  gsNS bfNS hNS; do
#    NAMENS="-$ns"
#    EXP_NAME=$ns
#    for rf in 0; do # 1 2 3 4; do   # number of random features
#        NAMENRF="-nrf$rf"

        TRAIN_ENV="ContFlappyBird$NAMENL$NAMENS$NAMENRF-train-v0"
        echo $TRAIN_ENV
        TEST_ENV="ContFlappyBird-clip$NAMENL$NAMENS$NAMENRF-test-v0"
        echo $TEST_ENV

        for MTHD in RNDM; do
#        for MTHD in A2C LSTM_A2C GRU_A2C DQN LSTM_DQN GRU_DQN PPO LSTM_PPO GRU_PPO; do #RNDM
            DIR_NAME="${MTHD}_NRF$rf"
            echo $EXP_NAME $DIR_NAME
            if [ $MTHD == A2C ]; then
                ARCHITECTURE='ff'
                BATCH_SIZE=64

                # SMAC config 1
                ACTIV_FCN='mixed'
                GAMMA=0.94
                ENT_COEFF=0.000036
                VF_COEFF=0.36
                LR=0.0032
                P_LAYER=17
                S_LAYER1=78
                S_LAYER2=35

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == LSTM_A2C ]; then
                ARCHITECTURE='lstm'
                BATCH_SIZE=64

                # SMAC config 3
                ACTIV_FCN='mixed'
                GAMMA=0.64
                ENT_COEFF=0.00007
                VF_COEFF=0.01
                LR=0.00088
                P_LAYER=47
                S_LAYER1=64
                S_LAYER2=22

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == GRU_A2C ]; then
                ARCHITECTURE='gru'
                BATCH_SIZE=64

                # SMAC config 4
                ACTIV_FCN='elu'
                GAMMA=0.8
                ENT_COEFF=0.00002
                VF_COEFF=0.2
                LR=0.0017
                P_LAYER=79
                S_LAYER1=14
                S_LAYER2=35

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == DQN ]; then
                ARCHITECTURE='ff'
                BUFFER_SIZE=500
                UPDATE_INTERVAL=64

                # BOHB config LOW / SMAC config 3
                ACTIV_FCN='elu'
                GAMMA=0.82
                EPS=0.35
                EPS_DECAY=0.990
                TAU=0.824
                LR=0.006
                LAYER1=39
                LAYER2=63
                LAYER3=73
                BATCH_SIZE=64

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == LSTM_DQN ]; then
                ARCHITECTURE='lstm'
                BUFFER_SIZE=500
                UPDATE_INTERVAL=64

                # SMAC config 1
                ACTIV_FCN='relu6'
                GAMMA=0.85
                EPS=0.22
                EPS_DECAY=0.975
                TAU=0.78
                LR=0.001
                LAYER1=71
                LAYER2=65
                LAYER3=85
                BATCH_SIZE=5
                TRACE_LENGTH=8

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE $TRACE_LENGTH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == GRU_DQN ]; then
                ARCHITECTURE='gru'
                BUFFER_SIZE=500
                UPDATE_INTERVAL=64

                # SMAC config 1
                ACTIV_FCN='relu6'
                GAMMA=0.85
                EPS=0.7
                EPS_DECAY=0.98
                TAU=0.86
                LR=0.002
                LAYER1=73
                LAYER2=34
                LAYER3=80
                BATCH_SIZE=5
                TRACE_LENGTH=8

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE $TRACE_LENGTH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == PPO ]; then
                ARCHITECTURE='ff'

                # SMAC configuration 1
                ACTIV_FCN='mixed'
                GAMMA=0.88
                ENT_COEFF=0.00007
                VF_COEFF=0.21
                LR=0.0042
                P_LAYER=21
                S_LAYER1=28
                S_LAYER2=59
                N_MINIBATCH=2
                N_OPTEPOCH=4
                N_STEPS=128

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == LSTM_PPO ]; then
                ARCHITECTURE='lstm'

                # SMAC Config 2 and 3 - combination of both
                ACTIV_FCN='relu6'
                GAMMA=0.9
                ENT_COEFF=0.00005
                VF_COEFF=0.3
                LR=0.001
                P_LAYER=24
                S_LAYER1=24
                S_LAYER2=24
                N_MINIBATCH=1
                N_OPTEPOCH=1
                N_STEPS=64

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == GRU_PPO ]; then
                ARCHITECTURE='gru'

                # SMAC config 3
                ACTIV_FCN='relu6'
                GAMMA=0.9
                ENT_COEFF=0.00001
                VF_COEFF=0.1
                LR=0.0015
                P_LAYER=24
                S_LAYER1=24
                S_LAYER2=24
                N_MINIBATCH=1
                N_OPTEPOCH=1
                N_STEPS=64

                ARGS=($ARCHITECTURE $TRAIN_ENV $TEST_ENV $TOTAL_TRAINSTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            elif [ $MTHD == RNDM ]; then
                NUM_AGENTS=1
                EVAL_MODEL='final'
                ARGS=($TEST_ENV $TOTAL_TRAINSTEPS $EVAL_MODEL)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_Training.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -e $EXP_NAME -d $DIR_NAME ${ARGS[@]}"

            fi
        done
    done
done