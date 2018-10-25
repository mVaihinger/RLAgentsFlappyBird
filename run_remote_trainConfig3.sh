#!/bin/bash
USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de
NUM_AGENTS=10

# Evaluate Hyperparameters found by SMAC. Run 10 training instances with each being initialized with a differnet random
# seed. While training, save intermediate models every 300th parameter update and evaluate it on a test environment
# which uses the random seed 3000.
# Fixed BAtch size, reduced buffer length, fixed learning rates

# TODO run LSTM_A2C, GRU_A2C with BOHB configs using higher minimum budget
# TODO run all methods with BOHB configs using lower minimum budget. (latest configs)

#HO_TYPE=BOHB
HO_TYPE=SMAC

MIN_BUDGET=LOW
#MIN_BUDGET=HIGH

# Sync data
chmod +x sync_code.sh
./sync_code.sh -u $USERHPC -h $HOST

TEST_INTERVAL=300 # save every n training updates.
TOTAL_TIMESTEPS=2000000
LOG_INTERVAL=300

#for MTHD in A2C DQN LSTM_A2C LSTM_DQN GRU_A2C GRU_DQN; do
for MTHD in PPO LSTM_PPO GRU_PPO; do  # PPO
#for MTHD in DQN LSTM_A2C LSTM_DQN GRU_A2C; do
#for MTHD in GRU_A2C GRU_DQN; do
#for MTHD in A2C; do
#for MTHD in DQN; do
#for MTHD in DRQN; do
#for MTHD in RA2C; do
    if [ $MTHD == A2C ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=20:00:00 # TODO
#        JOB_RESOURCES=nodes=1:ppn=10,pmem=2gb,walltime=10:00:00

        # SMAC and RLMETHOD params
        ARCHITECTURE='ff'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=9000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL=500 # every n training updates, so total_timesteps / (nenvs*nsteps*log_interval) = 80 logs during training

        NENVS=1 # TODO 10
        BATCH_SIZE=30

        if [ $HO_TYPE == SMAC ]; then

            # 4 best configurations
            a_ACTIV_FCN=('mixed' 'relu6' 'mixed' 'relu6')
            a_GAMMA=(0.9438 0.9649 0.9618 0.9308)
            a_ENT_COEFF=(0.000036 0.000021 0.000015 0.000084)
            a_VF_COEFF=(0.36288 0.37368 0.10205 0.17552)
#            a_LR=(0.01 0.01 0.01 0.01)
            a_LR=(0.0032 0.0013 0.0027 0.0041)
            a_P_LAYER=(17 17 34 75)
            a_S_LAYER1=(78 91 60 36)
            a_S_LAYER2=(35 51 66 76)

            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                ENT_COEFF=${a_ENT_COEFF[idx]}
                VF_COEFF=${a_VF_COEFF[idx]}
                LR=${a_LR[idx]}
                P_LAYER=${a_P_LAYER[idx]}
                S_LAYER1=${a_S_LAYER1[idx]}
                S_LAYER2=${a_S_LAYER2[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then

            # lower min budget
            if [ $MIN_BUDGET == LOW ]; then
                ACTIV_FCN='relu6'
                GAMMA=0.742
                ENT_COEFF=0.000039
                VF_COEFF=0.35869
                LR=0.00258
                P_LAYER=69
                S_LAYER1=79
                S_LAYER2=93
            elif [ $MIN_BUDGET == HIGH ]; then
                #higher min budget
                ACTIV_FCN='mixed'
                GAMMA=0.813
                ENT_COEFF=0.000066
                VF_COEFF=0.126
                LR=0.00586
                P_LAYER=39
                S_LAYER1=19
                S_LAYER2=85
            fi

            ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
        fi

    elif [ $MTHD == LSTM_A2C ]; then
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=00:30:00
        JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=20:00:00
        ARCHITECTURE='lstm'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=9000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL=500 # every n training updates

        NENVS=1 # 10
        BATCH_SIZE=30

        if [ $HO_TYPE == SMAC ]; then
            # 4 best configurations
            a_ACTIV_FCN=('relu6' 'relu6' 'mixed' 'elu')
            a_GAMMA=(0.9 0.7921 0.6421 0.7949)
            a_ENT_COEFF=(0.00001 0.000051 0.000073 0.000025)
            a_VF_COEFF=(0.1 0.32073 0.01046 0.19613)
            a_LR=(0.001 0.00031 0.00088 0.00174)
            a_P_LAYER=(24 29 47 79)
            a_S_LAYER1=(24 81 64 14)
            a_S_LAYER2=(24 22 22 35)

            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                ENT_COEFF=${a_ENT_COEFF[idx]}
                VF_COEFF=${a_VF_COEFF[idx]}
                LR=${a_LR[idx]}
                P_LAYER=${a_P_LAYER[idx]}
                S_LAYER1=${a_S_LAYER1[idx]}
                S_LAYER2=${a_S_LAYER2[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then
            if [ $MIN_BUDGET == LOW ]; then
                # lower min budget
                ACTIV_FCN='elu'
                GAMMA=0.897
                ENT_COEFF=0.000060
                VF_COEFF=0.032
                LR=0.00120
                P_LAYER=55
                S_LAYER1=74
                S_LAYER2=40
            elif [ $MIN_BUDGET == HIGH ]; then
                # higher min budget
                ACTIV_FCN='mixed'
                GAMMA=0.729
                ENT_COEFF=0.000099
                VF_COEFF=0.041
                LR=0.00808
                P_LAYER=97
                S_LAYER1=20
                S_LAYER2=72
            fi

            ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
        fi

    elif [ $MTHD == GRU_A2C ]; then
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=00:30:00
        JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=20:00:00

        ARCHITECTURE='gru'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=9000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL= # every n training updates

        NENVS=1 # 10
        BATCH_SIZE=30

        if [ $HO_TYPE == SMAC ]; then
            # 4 best configurations
            a_ACTIV_FCN=('mixed' 'elu' 'elu' 'elu')
            a_GAMMA=(0.720 0.615 0.760 0.795)
            a_ENT_COEFF=(0.00003 0.00007 0.00004 0.00002)
            a_VF_COEFF=(0.0497 0.4256 0.1501 0.1961)
            a_LR=(0.006 0.0033 0.0047 0.0017)
            a_P_LAYER=(84 61 39 79)
            a_S_LAYER1=(12 34 23 14)
            a_S_LAYER2=(84 19 53 35)

            # 4 best configurations with gru bug --< used lstm cell instead of gru cell
#            a_ACTIV_FCN=('relu6' 'relu6' 'elu' 'mixed')
#            a_GAMMA=(0.9 0.79216 0.81128 0.64211)
#            a_ENT_COEFF=(0.00001 0.000051 0.000073 0.000073)
#            a_VF_COEFF=(0.1 0.32073 0.08541 0.01046)
#            a_LR=(0.001 0.00031 0.00051 0.00088)
#            a_P_LAYER=(24 29 71 47)
#            a_S_LAYER1=(24 81 17 64)
#            a_S_LAYER2=(24 22 89 22)
            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                ENT_COEFF=${a_ENT_COEFF[idx]}
                VF_COEFF=${a_VF_COEFF[idx]}
                LR=${a_LR[idx]}
                P_LAYER=${a_P_LAYER[idx]}
                S_LAYER1=${a_S_LAYER1[idx]}
                S_LAYER2=${a_S_LAYER2[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then
            if [ $MIN_BUDGET == LOW ]; then
                #lower min budget
                ACTIV_FCN='elu'
                GAMMA=0.827
                ENT_COEFF=0.000039
                VF_COEFF=0.36215
                LR=0.00680
                P_LAYER=83
                S_LAYER1=65
                S_LAYER2=64
#                # with GRU bug
#                ACTIV_FCN='elu'
#                GAMMA=0.654
#                ENT_COEFF=0.000062
#                VF_COEFF=0.03788
#                LR=0.00090
#                P_LAYER=15
#                S_LAYER1=24
#                S_LAYER2=25
            elif [ $MIN_BUDGET == HIGH ]; then
                # higher min budget
                ACTIV_FCN='mixed'
                GAMMA=0.850
                ENT_COEFF=0.000024
                VF_COEFF=0.010
                LR=0.00096
                P_LAYER=63
                S_LAYER1=66
                S_LAYER2=10
            fi

            ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $BATCH_SIZE $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2)
            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
        fi

    elif [ $MTHD == DQN ]; then
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=00:30:00
        JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=20:00:00

        ARCHITECTURE='ff'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=1000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL=70 # every n training updates

        BUFFER_SIZE=500
        UPDATE_INTERVAL=30

        if [ $HO_TYPE == SMAC ]; then
            a_ACTIV_FCN=('mixed' 'elu' 'elu' 'mixed')
            a_GAMMA=(0.8628332696 0.7936504101 0.8172075803 0.8728128951)
            a_LR=(0.00950 0.00835 0.00600 0.00582)
            a_EPS=(0.268 0.875 0.351 0.622)
            a_EPS_DECAY=(0.955 0.966 0.990 0.307)
            a_TAU=(0.671 0.518 0.824 0.698)
            a_LAYER1=(53 92 39 100)
            a_LAYER2=(62 37 63 64)
            a_LAYER3=(79 94 73 48)
            a_BATCH_SIZE=(85 83 47 67)  # use recent interactions + buffer data

            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                EPS=${a_EPS[idx]}
                EPS_DECAY=${a_EPS_DECAY[idx]}
                TAU=${a_TAU[idx]}
                LR=${a_LR[idx]}
                LAYER1=${a_LAYER1[idx]}
                LAYER2=${a_LAYER2[idx]}
                LAYER3=${a_LAYER3[idx]}
                BATCH_SIZE=${a_BATCH_SIZE[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then
            if [ $MIN_BUDGET == LOW ]; then
                   #{'activ_fcn': 'mixed', 'batch_size': 92, 'epsilon': 0.7277555555882191, 'epsilon_decay': 0.8049263562381186, 'gamma': 0.7910148974322168, 'lr': 0.003036094030358841, 'tau': 0.9208141916782364, 'units_layer1': 56, 'units_layer2': 95, 'units_layer3': 71}

                ACTIV_FCN='mixed'
                GAMMA=0.791
                EPS=0.728
                EPS_DECAY=0.805
                TAU=0.921
                LR=0.00304
                LAYER1=56
                LAYER2=95
                LAYER3=71
                BATCH_SIZE=92
            elif [ $MIN_BUDGET == HIGH ]; then
                ACTIV_FCN='elu'
                GAMMA=0.877
                EPS=0.260
                EPS_DECAY=0.507
                TAU=0.631
                LR=0.00361
                LAYER1=100
                LAYER2=100
                LAYER3=23
                BATCH_SIZE=39
            fi

            ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE)
            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
        fi

    elif [ $MTHD == LSTM_DQN ]; then
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=00:30:00
        JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=20:00:00

        ARCHITECTURE='lstm'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=1000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL=70 # every n training updates

        BUFFER_SIZE=500
        UPDATE_INTERVAL=30
        BATCH_SIZE=5 # so gradient is updated based on 12*30 = 360 samples each time

        if [ $HO_TYPE == SMAC ]; then

            # Optimized parameters
            a_ACTIV_FCN=('relu6' 'mixed' 'mixed' 'relu6')
            a_GAMMA=(0.850319709 0.8774492995 0.8675073307 0.8518468276)
            a_LR=(0.00101 0.00151 0.00199 0.00362)
            a_EPS=(0.221 0.561 0.270 0.517)
            a_EPS_DECAY=(0.975 0.674 0.646 0.471)
            a_TAU=(0.783 0.676 0.603 0.643)
            a_LAYER1=(71 60 67 13)
            a_LAYER2=(65 40 26 21)
            a_LAYER3=(85 60 55 75)
            a_TRACE_LENGTH=(7 6 11 10)  # use recent interactions + buffer data

            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                EPS=${a_EPS[idx]}
                EPS_DECAY=${a_EPS_DECAY[idx]}
                TAU=${a_TAU[idx]}
                LR=${a_LR[idx]}
                LAYER1=${a_LAYER1[idx]}
                LAYER2=${a_LAYER2[idx]}
                LAYER3=${a_LAYER3[idx]}
                TRACE_LENGTH=${a_TRACE_LENGTH[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE $TRACE_LENGTH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then
            if [ $MIN_BUDGET == LOW ]; then
                ACTIV_FCN='relu6'
                GAMMA=0.777
                EPS=0.775
                EPS_DECAY=0.457
                TAU=0.552
                LR=0.00453
                LAYER1=15
                LAYER2=96
                LAYER3=69
                TRACE_LENGTH=9
            elif [ $MIN_BUDGET == HIGH ]; then
                ACTIV_FCN='elu'
                GAMMA=0.717
                EPS=0.639
                EPS_DECAY=0.670
                TAU=0.582
                LR=0.00170
                LAYER1=32
                LAYER2=40
                LAYER3=32
                TRACE_LENGTH=7
            fi

            ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE $TRACE_LENGTH)
            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
        fi

    elif [ $MTHD == GRU_DQN ]; then
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=00:30:00
        JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=20:00:00

        ARCHITECTURE='gru'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=1000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL=70 # every n training updates

        BUFFER_SIZE=500
        UPDATE_INTERVAL=30
        BATCH_SIZE=5 # so gradient is updated based on 12*30 = 360 samples each time

#        7	-66.3	relu6	0.6978189564	0.9813060463	0.8501040662	0.0020212232	0.8546718968	10	73	34	80
#        8	-46.95	mixed	0.588122088	    0.9670379563	0.8454106632	0.0017194249	0.5495825854	10	48	45	59
#        19	-46.4	relu6	0.7145951348	0.4831491665	0.8213083535	0.0028086198	0.6166516245	7	35	37	73
#        1	-43	    mixed	0.2273751008	0.3017491092	0.7727420124	0.0038855202	0.5662043079	9	28	40	85

        if [ $HO_TYPE == SMAC ]; then
            a_ACTIV_FCN=('relu6' 'mixed' 'relu6' 'mixed')
            a_GAMMA=(0.850 0.845 0.821 0.773)
            a_LR=(0.00202 0.00172 0.00281 0.00389)
            a_EPS=(0.698 0.588 0.715 0.227)
            a_EPS_DECAY=(0.981 0.967 0.483 0.302)
            a_TAU=(0.855 0.550 0.617 0.566)
            a_LAYER1=(73 48 35 28)
            a_LAYER2=(34 45 37 40)
            a_LAYER3=(80 59 73 85)
            a_TRACE_LENGTH=(10 10 7 9)  # use recent interactions + buffer data

            # 4 best configurations with gru bug --< used lstm cell instead of gru cell
#            a_ACTIV_FCN=('mixed' 'relu6' 'relu6' 'relu6')
#            a_GAMMA=(0.8171429957 0.7257616622 0.8631958665 0.7141693529)
#            a_LR=(0.00324 0.00209 0.00665 0.00226)
#            a_EPS=(0.230 0.533 0.622 0.707)
#            a_EPS_DECAY=(0.557 0.996 0.838 0.220)
#            a_TAU=(0.943 0.771 0.804 0.602)
#            a_LAYER1=(13 68 16 45)
#            a_LAYER2=(43 94 71 37)
#            a_LAYER3=(88 44 35 32)
#            a_TRACE_LENGTH=(7 5 8 14)  # use recent interactions + buffer data

            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                EPS=${a_EPS[idx]}
                EPS_DECAY=${a_EPS_DECAY[idx]}
                TAU=${a_TAU[idx]}
                LR=${a_LR[idx]}
                LAYER1=${a_LAYER1[idx]}
                LAYER2=${a_LAYER2[idx]}
                LAYER3=${a_LAYER3[idx]}
                TRACE_LENGTH=${a_TRACE_LENGTH[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE $TRACE_LENGTH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then
            if [ $MIN_BUDGET == LOW ]; then
                ACTIV_FCN='mixed'
                GAMMA=0.844
                EPS=0.480
                EPS_DECAY=0.566
                TAU=0.600
                LR=0.00138
                LAYER1=82
                LAYER2=20
                LAYER3=88
                TRACE_LENGTH=10
            elif [ $MIN_BUDGET == HIGH ]; then
                ACTIV_FCN='relu6'
                GAMMA=0.844
                EPS=0.823
                EPS_DECAY=0.803
                TAU=0.943
                LR=0.00183
                LAYER1=49
                LAYER2=95
                LAYER3=20
                TRACE_LENGTH=7
            fi

            ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $BUFFER_SIZE $UPDATE_INTERVAL $ACTIV_FCN $GAMMA $EPS $EPS_DECAY $TAU $LR $LAYER1 $LAYER2 $LAYER3 $BATCH_SIZE $TRACE_LENGTH)
            echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
            ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
        fi
    elif [ $MTHD == PPO ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=20:00:00 # TODO
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=00:30:00

        # SMAC and RLMETHOD params
        ARCHITECTURE='ff'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=9000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL=500 # every n training updates, so total_timesteps / (nenvs*nsteps*log_interval) = 80 logs during training

        NENVS=1 # TODO 10
#        BATCH_SIZE=30

        if [ $HO_TYPE == SMAC ]; then

            # 4 best configurations
            a_ACTIV_FCN=(mixed relu6 mixed elu)
            a_GAMMA=(0.8756 0.9268 0.8946 0.9058)
            a_ENT_COEFF=(0.0000709 0.0000945 0.0000505 0.0000104)
            a_VF_COEFF=(0.214 0.110 0.281 0.094)
#            a_LR=(0.01 0.01 0.01 0.01)
            a_LR=(0.00420 0.00559 0.00808 0.00181)
            a_P_LAYER=(21 46 87 76)
            a_S_LAYER1=(28 58 83 86)
            a_S_LAYER2=(59 48 50 89)
            a_N_MINIBATCH=(2 4 2 8)
            a_N_OPTEPOCH=(4 7 3 2)
            a_N_STEPS=(128 128 256 128)

            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                ENT_COEFF=${a_ENT_COEFF[idx]}
                VF_COEFF=${a_VF_COEFF[idx]}
                LR=${a_LR[idx]}
                P_LAYER=${a_P_LAYER[idx]}
                S_LAYER1=${a_S_LAYER1[idx]}
                S_LAYER2=${a_S_LAYER2[idx]}
                N_MINIBATCH=${a_N_MINIBATCH[idx]}
                N_OPTEPOCH=${a_N_OPTEPOCH[idx]}
                N_STEPS=${a_N_STEPS[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then

            # lower min budget
            if [ $MIN_BUDGET == LOW ]; then
                ACTIV_FCN='elu'
                GAMMA=0.89
                ENT_COEFF=0.000046
                VF_COEFF=0.27
                LR=0.0045
                P_LAYER=10
                S_LAYER1=71
                S_LAYER2=82
                N_MINIBATCH=4
                N_OPTEPOCH=7
                N_STEPS=128

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            fi
        fi
    elif [ $MTHD == LSTM_PPO ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=20:00:00 # TODO
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=00:30:00

        # SMAC and RLMETHOD params
        ARCHITECTURE='lstm'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=9000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL=500 # every n training updates, so total_timesteps / (nenvs*nsteps*log_interval) = 80 logs during training

        NENVS=1 # TODO 10
#        BATCH_SIZE=30

        if [ $HO_TYPE == SMAC ]; then

            # 4 best configurations
            a_ACTIV_FCN=(relu6 relu6 mixed relu6)
            a_GAMMA=(0.737 0.9 0.899 0.708)
            a_ENT_COEFF=(0.000083 0.00001 0.000091 0.000045)
            a_VF_COEFF=(0.242 0.1 0.359 0.479)
#            a_LR=(0.01 0.01 0.01 0.01)
            a_LR=(0.00689 0.001 0.00117 0.00107)
            a_P_LAYER=(92 24 43 16)
            a_S_LAYER1=(45 24 19 12)
            a_S_LAYER2=(56 24 41 34)
            a_N_MINIBATCH=(1 1 1 1)
            a_N_OPTEPOCH=(1 1 1 1)
            a_N_STEPS=(64 64 64 64)

            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                ENT_COEFF=${a_ENT_COEFF[idx]}
                VF_COEFF=${a_VF_COEFF[idx]}
                LR=${a_LR[idx]}
                P_LAYER=${a_P_LAYER[idx]}
                S_LAYER1=${a_S_LAYER1[idx]}
                S_LAYER2=${a_S_LAYER2[idx]}
                N_MINIBATCH=${a_N_MINIBATCH[idx]}
                N_OPTEPOCH=${a_N_OPTEPOCH[idx]}
                N_STEPS=${a_N_STEPS[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then

            # lower min budget
            if [ $MIN_BUDGET == LOW ]; then
                ACTIV_FCN='relu6'
                GAMMA=0.669
                ENT_COEFF=0.000064
                VF_COEFF=0.1575
                LR=0.0054
                P_LAYER=70
                S_LAYER1=91
                S_LAYER2=65
                N_MINIBATCH=1
                N_OPTEPOCH=1
                N_STEPS=64

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            fi
        fi
    elif [ $MTHD == GRU_PPO ]; then
        JOB_RESOURCES=nodes=1:ppn=2,pmem=1gb,walltime=20:00:00 # TODO
#        JOB_RESOURCES=nodes=1:ppn=2,pmem=2gb,walltime=00:30:00

        # SMAC and RLMETHOD params
        ARCHITECTURE='gru'
        ENV='ContFlappyBird-v1'  # v1 - stationary, non-clipped episodes
                             # v2 - non-stationary, non-clipped episodes
                             # v3 - stationary, clipped episodes
                             # v4 - non-stationary, non-clipped episodes
        TEST_ENV='ContFlappyBird-v3'
#        TOTAL_TIMESTEPS=9000000
        MAX_GRAD_NORM=0.01
#        LOG_INTERVAL=500 # every n training updates, so total_timesteps / (nenvs*nsteps*log_interval) = 80 logs during training

        NENVS=1 # TODO 10
#        BATCH_SIZE=30

        if [ $HO_TYPE == SMAC ]; then

            # 4 best configurations
            a_ACTIV_FCN=(relu6 mixed relu6 relu6)
            a_GAMMA=(0.903 0.932 0.9 0.691)
            a_ENT_COEFF=(0.000049 0.000055 0.00001 0.000017)
            a_VF_COEFF=(0.430 0.083 0.1 0.28)
#            a_LR=(0.01 0.01 0.01 0.01)
            a_LR=(0.00393 0.00240 0.00100 0.00150)
            a_P_LAYER=(52 45 24 84)
            a_S_LAYER1=(40 82 24 64)
            a_S_LAYER2=(84 41 24  14)
            a_N_MINIBATCH=(1 1 1 1)
            a_N_OPTEPOCH=(1 1 1 1)
            a_N_STEPS=(32 128 64 32)

            for ((idx=0; idx<4; ++idx)); do
                ACTIV_FCN=${a_ACTIV_FCN[idx]}
                GAMMA=${a_GAMMA[idx]}
                ENT_COEFF=${a_ENT_COEFF[idx]}
                VF_COEFF=${a_VF_COEFF[idx]}
                LR=${a_LR[idx]}
                P_LAYER=${a_P_LAYER[idx]}
                S_LAYER1=${a_S_LAYER1[idx]}
                S_LAYER2=${a_S_LAYER2[idx]}
                N_MINIBATCH=${a_N_MINIBATCH[idx]}
                N_OPTEPOCH=${a_N_OPTEPOCH[idx]}
                N_STEPS=${a_N_STEPS[idx]}

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            done
        elif [ $HO_TYPE == BOHB ]; then

            # lower min budget
            if [ $MIN_BUDGET == LOW ]; then
                ACTIV_FCN='mixed'
                GAMMA=0.739
                ENT_COEFF=0.000020
                VF_COEFF=0.157
                LR=0.0015
                P_LAYER=33
                S_LAYER1=32
                S_LAYER2=86
                N_MINIBATCH=1
                N_OPTEPOCH=1
                N_STEPS=64

                ARGS=($ARCHITECTURE $ENV $TEST_ENV $TOTAL_TIMESTEPS $MAX_GRAD_NORM $LOG_INTERVAL $TEST_INTERVAL $NENVS $N_STEPS $ACTIV_FCN $GAMMA $ENT_COEFF $VF_COEFF $LR $P_LAYER $S_LAYER1 $S_LAYER2 $N_MINIBATCH $N_OPTEPOCH)
                echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
                ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES -o $HO_TYPE ${ARGS[@]}"
            fi
        fi
    fi
#    echo -e "\nRunning "$NUM_AGENTS" learning agents using "$MTHD" algorithm with optimized configuration"
#    ssh $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_TrainConfig.sh -r $NUM_AGENTS -m $MTHD -l $JOB_RESOURCES ${ARGS[@]}"
done
