#!/bin/bash
USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de
NUM_WORKERS=20

# Sync data
chmod +x sync_code.sh
./sync_code.sh -u $USERHPC -h $HOST

# UPDATE_INTERVAL = A2C_BATCH_SIZE = 30

#MIN_STEPS=32
#MAX_STEPS=1024

#for MTHD in A2C DQN LSTM_A2C LSTM_DQN GRU_A2C GRU_DQN; do
for MTHD in GRU_DQN GRU_A2C; do
#for MTHD in A2C DQN LSTM_DQN; do
#for MTHD in DQN; do  #  LSTM_DQN GRU_DQN; do
    if [ $MTHD == A2C ]; then
        JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='ff'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        MIN_STEPS=10000
        MAX_STEPS=3000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=100 # every n training updates
        EVAL_MODEL='all'

        NENVS=10
        BATCH_SIZE=30

#        NENVS=3
#        BATCH_SIZE=50
#        LR=0.0005
#        GAMMA=0.90
#        VF_COEFF=0.2
#        ENT_COEFF=0.0000001
#        LAYER1=64
#        LAYER2=64
#        LAYER3=64


        ARGS=($ARCHITECTURE $ENV $TEST_ENV $MIN_STEPS $MAX_STEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS $BATCH_SIZE)
    elif [ $MTHD == LSTM_A2C ]; then
        JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='lstm'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        MIN_STEPS=10000
        MAX_STEPS=3000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=100 # every n training updates
        EVAL_MODEL='all'

        NENVS=10
        BATCH_SIZE=30

#        NENVS=3
#        BATCH_SIZE=50
#        LR=0.0005
#        GAMMA=0.90
#        VF_COEFF=0.2
#        ENT_COEFF=0.0000001
#        LAYER1=64
#        LAYER2=64
#        LAYER3=64

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $MIN_STEPS $MAX_STEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS $BATCH_SIZE)
    elif [ $MTHD == GRU_A2C ]; then
        JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='gru'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        MIN_STEPS=10000
        MAX_STEPS=9000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=100 # every n training updates
        EVAL_MODEL='all'

        NENVS=10
        BATCH_SIZE=30

#        NENVS=3
#        BATCH_SIZE=50
#        LR=0.0005
#        GAMMA=0.90
#        VF_COEFF=0.2
#        ENT_COEFF=0.0000001
#        LAYER1=64
#        LAYER2=64
#        LAYER3=64

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $MIN_STEPS $MAX_STEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS $BATCH_SIZE)
    elif [ $MTHD == DQN ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='ff'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        MIN_STEPS=5000
        MAX_STEPS=1000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=100 # every n training updates
        EVAL_MODEL='all'

        BUFFER_SIZE=500
        UPDATE_INTERVAL=30

#        GAMMA=0.90
#        EPSILON=0.5
#        EPS_DECAY=0.995
#        TAU=0.90
#        LR=0.0005
#        BATCH_SIZE=30
#        TRACE_LENGTH=1 # is reset anyways
#        LAYER1=64
#        LAYER2=64
#        LAYER3=64

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $MIN_STEPS $MAX_STEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $BUFFER_SIZE $UPDATE_INTERVAL)  #  $GAMMA $EPSILON $EPS_DECAY $TAU $LR $TRACE_LENGTH $LAYER1 $LAYER2 $LAYER3 $ACTIV_FCN)
    elif [ $MTHD == LSTM_DQN ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='lstm'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        MIN_STEPS=5000
        MAX_STEPS=1000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=100 # every n training updates
        EVAL_MODEL='all'

        BUFFER_SIZE=500
        UPDATE_INTERVAL=30
        BATCH_SIZE=5 # so gradient is updated based on 12*30 = 360 samples each time


        ARGS=($ARCHITECTURE $ENV $TEST_ENV $MIN_STEPS $MAX_STEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $BUFFER_SIZE $BATCH_SIZE $UPDATE_INTERVAL)  #  $GAMMA $EPSILON $EPS_DECAY $TAU $LR $TRACE_LENGTH $LAYER1 $LAYER2 $LAYER3 $ACTIV_FCN)
    elif [ $MTHD == GRU_DQN ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='gru'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        MIN_STEPS=5000
        MAX_STEPS=5000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=100 # every n training updates
        EVAL_MODEL='all'

        BUFFER_SIZE=500
        UPDATE_INTERVAL=30
        BATCH_SIZE=5 # so gradient is updated based on 12*30 = 360 samples each time


        ARGS=($ARCHITECTURE $ENV $TEST_ENV $MIN_STEPS $MAX_STEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $BUFFER_SIZE $BATCH_SIZE $UPDATE_INTERVAL)  #  $GAMMA $EPSILON $EPS_DECAY $TAU $LR $TRACE_LENGTH $LAYER1 $LAYER2 $LAYER3 $ACTIV_FCN)
    fi

    echo -e "\nRunning "$NUM_WORKERS" bohb workers optimizing "$MTHD" algorithm configuration"
    ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_BOHB_workers.sh -w $NUM_WORKERS -m $MTHD -l $JOB_RESOURCES ${ARGS[@]}"
done
