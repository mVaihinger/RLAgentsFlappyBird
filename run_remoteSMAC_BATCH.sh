#!/bin/bash
USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de
RUN_JOBS=30
RUNCOUNT_LIMIT=40  # limit of algorithm evaluations of a smac instance
#JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=20:00:00

# Sync data
chmod +x sync_code.sh
./sync_code.sh -u $USERHPC -h $HOST

#for MTHD in A2C DQN DRQN RA2C; do
#for MTHD in A2C; do
#for MTHD in DQN; do
for MTHD in DRQN; do
#for MTHD in A2C RA2C; do
    if [ $MTHD == A2C ]; then
        JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=24:00:00

        # SMAC and RLMETHOD params
        ENV='FlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        POLICY='largemlp'
        TOTAL_TIMESTEPS=2000000
        NENVS=10
        GAMMA=0.90
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=50 # every n training updates, so total_timesteps / (nenvs*nsteps*log_interval) = 80 logs
        SHOW_INTERVAL=0 # no rendering
        TEST_INTERVAL=0 # no testing during learning

        NSTEPS=37  # so nbatch = 370
        LR=0.00016

        ARGS=($RUNCOUNT_LIMIT $ENV $POLICY $TOTAL_TIMESTEPS $MAX_GRAD_NORM $NENVS $GAMMA $LOG_INTERVAL $SHOW_INTERVAL $TEST_INTERVAL $NSTEPS $LR)
    elif [ $MTHD == DQN ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=28:00:00
        ENV='FlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TOTAL_TIMESTEPS=500000
        MAX_GRAD_NORM=0.01
        BUFFER_SIZE=2000 # 5000
        UPDATE_INTERVAL=100 #
        LOG_INTERVAL=3000  # total_timesteps / (log_interval) = 100 logs
        SHOW_INTERVAL=0 # no rendering
        TEST_INTERVAL=0 # no testing during learning

        GAMMA=0.85
        LR=0.0015
        BATCH_SIZE=350

        ARGS=($RUNCOUNT_LIMIT $ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $BUFFER_SIZE $LOG_INTERVAL $SHOW_INTERVAL $TEST_INTERVAL $GAMMA $LR $BATCH_SIZE)
    elif [ $MTHD == DRQN ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=28:00:00 # TODO
        ENV='FlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TOTAL_TIMESTEPS=500000
        TAU=0.99
        MAX_GRAD_NORM=0.01
        BUFFER_SIZE=100
        UPDATE_INTERVAL=100
        LOG_INTERVAL=1000  # total_timesteps / (update_interval * log_interval) = 50 logs
        SHOW_INTERVAL=0 # no rendering
        TEST_INTERVAL=0 # no testing during learning

        GAMMA=0.90
        LR=0.0005
        TRACE_LENGTH=12
        NBATCH=30  # so gradient is updated based on 12*30 = 360 samples each time


        ARGS=($RUNCOUNT_LIMIT $ENV $TOTAL_TIMESTEPS $TAU $MAX_GRAD_NORM $BUFFER_SIZE $UPDATE_INTERVAL $LOG_INTERVAL $SHOW_INTERVAL $TEST_INTERVAL $GAMMA $LR $TRACE_LENGTH $NBATCH)
    elif [ $MTHD == RA2C ]; then
        JOB_RESOURCES=nodes=1:ppn=10,pmem=1gb,walltime=24:00:00

        # SMAC and RLMETHOD params
        ENV='FlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        POLICY='lstm'
        TOTAL_TIMESTEPS=2000000
        NENVS=10
        GAMMA=0.90
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=50 # every n training updates
        SHOW_INTERVAL=0 # no rendering
        TEST_INTERVAL=0 # no testing during learning

        NSTEPS=21
        LR=0.00065
        ARGS=($RUNCOUNT_LIMIT $ENV $POLICY $TOTAL_TIMESTEPS $MAX_GRAD_NORM $NENVS $GAMMA $LOG_INTERVAL $SHOW_INTERVAL $TEST_INTERVAL $NSTEPS $LR)
    fi

    echo -e "\nRunning "$RUN_JOBS" smac instances optimizing "$MTHD" algorithm configuration"
    ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_SMAC_BATCH.sh -r $RUN_JOBS -m $MTHD -l $JOB_RESOURCES ${ARGS[@]}"
done
