#!/bin/bash
USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de
RUN_JOBS=25
RUNCOUNT_LIMIT=40  # limit of algorithm evaluations of a smac instance
JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=30:00:00

# Sync data
chmod +x ../sync_code.sh
../sync_code.sh -u $USERHPC -h $HOST

# UPDATE_INTERVAL = A2C_BATCH_SIZE = 30
#JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=00:30:00
TOTAL_TIMESTEPS=2000000
EVAL_MODEL='inter'
#for MTHD in A2C DQN LSTM_A2C LSTM_DQN GRU_A2C GRU_DQN; do
for MTHD in LSTM_PPO GRU_PPO; do
    echo $MTHD
#for MTHD in DQN; do
    if [ $MTHD == A2C ]; then
        #JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='ff'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        #TOTAL_TIMESTEPS=1500000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=70 # every n training updates

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


        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS $BATCH_SIZE)
    elif [ $MTHD == LSTM_A2C ]; then
        JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='lstm'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        TOTAL_TIMESTEPS=1000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=70 # every n training updates

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

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS $BATCH_SIZE)
    elif [ $MTHD == GRU_A2C ]; then
        JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=24:00:00
        ARCHITECTURE='gru'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        TOTAL_TIMESTEPS=1000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=70 # every n training updates

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

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS $BATCH_SIZE)
    elif [ $MTHD == DQN ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=28:00:00
        ARCHITECTURE='ff'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        TOTAL_TIMESTEPS=200000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=70 # every n training updates

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

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $BUFFER_SIZE $UPDATE_INTERVAL)  #  $GAMMA $EPSILON $EPS_DECAY $TAU $LR $TRACE_LENGTH $LAYER1 $LAYER2 $LAYER3 $ACTIV_FCN)
    elif [ $MTHD == LSTM_DQN ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=28:00:00
        ARCHITECTURE='lstm'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        TOTAL_TIMESTEPS=200000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=70 # every n training updates

        BUFFER_SIZE=500
        UPDATE_INTERVAL=30
        BATCH_SIZE=5 # so gradient is updated based on 12*30 = 360 samples each time


        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $BUFFER_SIZE $BATCH_SIZE $UPDATE_INTERVAL)  #  $GAMMA $EPSILON $EPS_DECAY $TAU $LR $TRACE_LENGTH $LAYER1 $LAYER2 $LAYER3 $ACTIV_FCN)
    elif [ $MTHD == GRU_DQN ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=28:00:00
        ARCHITECTURE='gru'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
        TOTAL_TIMESTEPS=200000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=70 # every n training updates

        BUFFER_SIZE=500
        UPDATE_INTERVAL=30
        BATCH_SIZE=5 # so gradient is updated based on 12*30 = 360 samples each time


        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $BUFFER_SIZE $BATCH_SIZE $UPDATE_INTERVAL)  #  $GAMMA $EPSILON $EPS_DECAY $TAU $LR $TRACE_LENGTH $LAYER1 $LAYER2 $LAYER3 $ACTIV_FCN)
    elif [ $MTHD == PPO ]; then
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=28:00:00
        ARCHITECTURE='ff'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=2000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=70 # every n training updates
        NENVS=1

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS)

    elif [ $MTHD == LSTM_PPO ]; then
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=28:00:00
        ARCHITECTURE='lstm'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=2000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=200 # every n training updates
        NENVS=1

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS)

    elif [ $MTHD == GRU_PPO ]; then
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=28:00:00
        ARCHITECTURE='gru'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=2000000
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=200 # every n training updates
        NENVS=1

        ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $EVAL_MODEL $NENVS)
    fi

    echo -e "\nRunning "$RUN_JOBS" smac instances optimizing "$MTHD" algorithm configuration"
    echo $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/HPO_scripts/run_SMAC_BATCH.sh -j $RUN_JOBS -r $RUNCOUNT_LIMIT -m $MTHD -l $JOB_RESOURCES ${ARGS[@]}"
    ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/HPO_scripts/run_SMAC_BATCH.sh -j $RUN_JOBS -r $RUNCOUNT_LIMIT -m $MTHD -l $JOB_RESOURCES ${ARGS[@]}"
done
