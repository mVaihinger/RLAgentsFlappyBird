#!/bin/bash
USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de
JOB_RESOURCES=nodes=1:ppn=1,pmem=400mb,walltime=01:00:00

# Sync data
chmod +x sync_code.sh
./sync_code.sh -u $USERHPC -h $HOST

#for MTHD in A2C DQN; do
#for MTHD in A2C; do
for MTHD in DQN; do
#for MTHD in DRQN; do
    if [ $MTHD == A2C ]; then
        # SMAC and RLMETHOD params
        ENV='FlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TOTAL_TIMESTEPS=2000000
        NENVS=3
        GAMMA=0.90
        MAX_GRAD_NORM=0.01
        LOG_INTERVAL=20 # every n training updates
        NSTEPS=50
        ARGS=($ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $NENVS $GAMMA $LOG_INTERVAL $NSTEPS)
    elif [ $MTHD == DQN ]; then
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, clipped episodes
        TEST_ENV='ContFlappyBird-v1'
        TOTAL_TIMESTEPS=90000
        MAX_GRAD_NORM=0.01
        BUFFER_SIZE=500
        LOG_INTERVAL=20
        ACTIV_FCN='relu6'
        ARGS=($ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $BUFFER_SIZE $ACTIV_FCN $LOG_INTERVAL $TEST_ENV)
    elif [ $MTHD == DRQN ]; then
        ENV='FlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TOTAL_TIMESTEPS=90000
        TAU=0.99
        MAX_GRAD_NORM=0.01
        BUFFER_SIZE=50
        ARGS=($ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $BUFFER_SIZE)
    fi

    echo -e "\nLearning to control env "$ENV" using "$MTHD" algorithm."
    ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_RLmsub.sh -m $MTHD -l $JOB_RESOURCES ${ARGS[@]}"
done
