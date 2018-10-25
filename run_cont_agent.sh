#!/bin/bash
if [ -n "${SCRIPT_FLAGS}" ] ; then
	if [ -z "${*}" ]; then
		set -- ${SCRIPT_FLAGS}
	fi
fi
MTHD=${1}
LOGDIR=${2}
DATE=${3}
JOB_IDX=${4}

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "Job id:                               $MOAB_JOBID"
echo "Job name:                             $MOAB_JOBNAME"
echo "Number of nodes allocated to job:     $MOAB_NODECOUNT"
echo "Number of cores allocated to job:     $MOAB_PROCCOUNT"

echo -e "\nchanging directory"
cd /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/  # $MTHD/
module load tools/singularity/2.6

if [ "$MTHD" == "A2C" ] || [ "$MTHD" == "LSTM_A2C" ] || [ "$MTHD" == "GRU_A2C" ]; then
    RUN_SCRIPT=run_a2c.py

    # VARS
    ARCHITECTURE=${5}
    ENV=${6}
    TEST_ENV=${7}
    TOTAL_TIMESTEPS=${8}
    MAX_GRAD_NORM=${9}
    LOG_INTERVAL=${10}
    TEST_INTERVAL=${11}

    NENVS=${12}
    BATCH_SIZE=${13}
    ACTIV_FCN=${14}
    GAMMA=${15}
    ENT_COEFF=${16}
    VF_COEFF=${17}
    LR=${18}
    P_LAYER=${19}
    S_LAYER1=${20}
    S_LAYER2=${21}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/A2C/$RUN_SCRIPT --logdir $LOGDIR --seed $JOB_IDX --keep_model 2 --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --test_interval $TEST_INTERVAL --nenvs $NENVS --batch_size $BATCH_SIZE --activ_fcn $ACTIV_FCN --gamma $GAMMA --ent_coeff $ENT_COEFF --vf_coeff $VF_COEFF --lr $LR --units_shared_layer1 $S_LAYER1 --units_shared_layer2 $S_LAYER2 --units_policy_layer $P_LAYER"
elif [ "$MTHD" == "DQN" ] || [ "$MTHD" == "LSTM_DQN" ] || [ "$MTHD" == "GRU_DQN" ]; then
    RUN_SCRIPT=run_dqn.py

    # VARS
    ARCHITECTURE=${5}
    ENV=${6}
    TEST_ENV=${7}
    TOTAL_TIMESTEPS=${8}
    MAX_GRAD_NORM=${9}
    LOG_INTERVAL=${10}
    TEST_INTERVAL=${11}

    BUFFER_SIZE=${12}
    UPDATE_INTERVAL=${13}
    ACTIV_FCN=${14}
    GAMMA=${15}
    EPSILON=${16}
    EPS_DECAY=${17}
    TAU=${18}
    LR=${19}
    UNITS_LAYER1=${20}  # (shared1, shared2, policy)
    UNITS_LAYER2=${21}
    UNITS_LAYER3=${22}
    BATCH_SIZE=${23}

    if [ "$MTHD" == "LSTM_DQN" ] || [ "$MTHD" == "GRU_DQN" ]; then
        TRACE_LENGTH=${24}
        PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/DQN/$RUN_SCRIPT --logdir $LOGDIR --seed $JOB_IDX --keep_model 2 --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --test_interval $TEST_INTERVAL --buffer_size $BUFFER_SIZE --update_interval $UPDATE_INTERVAL --activ_fcn $ACTIV_FCN --gamma $GAMMA --epsilon $EPSILON --epsilon_decay $EPS_DECAY --tau $TAU --lr $LR --units_layer1 $UNITS_LAYER1 --units_layer2 $UNITS_LAYER2 --units_layer3 $UNITS_LAYER3 --batch_size $BATCH_SIZE --trace_length $TRACE_LENGTH"
    else
        PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/DQN/$RUN_SCRIPT --logdir $LOGDIR --seed $JOB_IDX --keep_model 2 --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --test_interval $TEST_INTERVAL --buffer_size $BUFFER_SIZE --update_interval $UPDATE_INTERVAL --activ_fcn $ACTIV_FCN --gamma $GAMMA --epsilon $EPSILON --epsilon_decay $EPS_DECAY --tau $TAU --lr $LR --units_layer1 $UNITS_LAYER1 --units_layer2 $UNITS_LAYER2 --units_layer3 $UNITS_LAYER3 --batch_size $BATCH_SIZE"
    fi
elif [ "$MTHD" == "PPO" ] || [ "$MTHD" == "LSTM_PPO" ] || [ "$MTHD" == "GRU_PPO" ]; then
    RUN_SCRIPT=run_ppo.py

    # VARS
    ARCHITECTURE=${5}
    ENV=${6}
    TEST_ENV=${7}
    TOTAL_TIMESTEPS=${8}
    MAX_GRAD_NORM=${9}
    LOG_INTERVAL=${10}
    TEST_INTERVAL=${11}

    NENVS=${12}
    BATCH_SIZE=${13}
    ACTIV_FCN=${14}
    GAMMA=${15}
    ENT_COEFF=${16}
    VF_COEFF=${17}
    LR=${18}
    P_LAYER=${19}
    S_LAYER1=${20}
    S_LAYER2=${21}
    N_MINIBATCH=${22}
    N_OPTEPOCH=${23}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/PPO/$RUN_SCRIPT --logdir $LOGDIR --seed $JOB_IDX --keep_model 2 --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --test_interval $TEST_INTERVAL --nenvs $NENVS --nsteps $BATCH_SIZE --activ_fcn $ACTIV_FCN --gamma $GAMMA --ent_coeff $ENT_COEFF --vf_coeff $VF_COEFF --lr $LR --units_shared_layer1 $S_LAYER1 --units_shared_layer2 $S_LAYER2 --units_policy_layer $P_LAYER --nminibatches $N_MINIBATCH --noptepochs $N_OPTEPOCH"
elif [ "$MTHD" == "RNDM" ]; then
    TEST_ENV=${5}
    TOTAL_TRAINSTEPS=${6}
    EVAL_MODEL=${7}
    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/RNDM/run_rndm.py --logdir $LOGDIR --seed $JOB_IDX --test_env $TEST_ENV --total_timesteps $TOTAL_TRAINSTEPS --eval_model $EVAL_MODEL"

else
    echo "method "$MTHD" is not known."
fi

echo "Run python command in singularity container"
echo $PYCMD
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_experiments_p36.img echo "Hola"
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_experiments_p36.img $PYCMD
