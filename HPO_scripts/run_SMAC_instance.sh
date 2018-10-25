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
RUNCOUNT_LIMIT=${5}

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "Job id:                               $MOAB_JOBID"
echo "Job name:                             $MOAB_JOBNAME"
echo "Number of nodes allocated to job:     $MOAB_NODECOUNT"
echo "Number of cores allocated to job:     $MOAB_PROCCOUNT"

echo -e "\nchanging directory"
cd /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/
module load tools/singularity/2.5

if [ "$MTHD" == "A2C" ] || [ "$MTHD" == "LSTM_A2C" ] || [ "$MTHD" == "GRU_A2C" ]; then
    echo -e "\nchanging directory"
    cd /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/A2C/

    RUN_SCRIPT=a2c_smac_wrapper.py

    # VARS
    ARCHITECTURE=${6}
    ENV=${7}
    TEST_ENV=${8}
    TOTAL_TIMESTEPS=${9}
    MAX_GRAD_NORM=${10}
    LOG_INTERVAL=${11}
    EVAL_MODEL=${12}

    NENVS=${13}
    BATCH_SIZE=${14}

#    ENV=${6}
#    POLICY=${7}
#    TOTAL_TIMESTEPS=${8}
#    MAX_GRAD_NORM=${9}
#    NENVS=${10}
#    GAMMA=${11}
#    LOG_INTERVAL=${12}
#    SHOW_INTERVAL=${13}
#    TEST_INTERVAL=${14}
#
#    NSTEPS=${15}
#    LR=${16}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/A2C/$RUN_SCRIPT --instance_id $JOB_IDX --run_parallel True --runcount_limit $RUNCOUNT_LIMIT --logdir $LOGDIR --seed $JOB_IDX --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --eval_model $EVAL_MODEL --nenvs $NENVS --batch_size $BATCH_SIZE"   # --gamma $GAMMA --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --nsteps $NSTEPS --lr $LR"
elif [ "$MTHD" == "DQN" ]; then
    echo -e "\nchanging directory"
    cd /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/DQN/
    RUN_SCRIPT=dqn_smac_wrapper.py

    ARCHITECTURE=${6}
    ENV=${7}
    TEST_ENV=${8}
    TOTAL_TIMESTEPS=${9}
    MAX_GRAD_NORM=${10}
    LOG_INTERVAL=${11}
    EVAL_MODEL=${12}

    BUFFER_SIZE=${13}
    UPDATE_INTERVAL=${14}

#    GAMMA=${11}
#    EPSILON=${12}
#    EPS_DECAY=${13}
#    TAU=${14}
#    LR=${15}
#    BATCH_SIZE=${17}
#    TRACE_LENGTH=${18}  # is reset anyways
#    LAYER1=${19}
#    LAYER2=${20}
#    LAYER3=${21}
#    ACTIV_FCN=${22}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/DQN/$RUN_SCRIPT --instance_id $JOB_IDX --run_parallel True --runcount_limit $RUNCOUNT_LIMIT --logdir $LOGDIR --seed $JOB_IDX --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --eval_model $EVAL_MODEL --buffer_size $BUFFER_SIZE --update_interval $UPDATE_INTERVAL"  #  --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --gamma $GAMMA --lr $LR"
elif [ "$MTHD" == "LSTM_DQN" ] || [ "$MTHD" == "GRU_DQN" ]; then
    echo -e "\nchanging directory"
    cd /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/DQN/

    RUN_SCRIPT=dqn_smac_wrapper.py

    ARCHITECTURE=${6}
    ENV=${7}
    TEST_ENV=${8}
    TOTAL_TIMESTEPS=${9}
    MAX_GRAD_NORM=${10}
    LOG_INTERVAL=${11}
    EVAL_MODEL=${12}

    BUFFER_SIZE=${13}
    UPDATE_INTERVAL=${14}
    BATCH_SIZE=${15}

#    GAMMA=${11}
#    EPSILON=${12}
#    EPS_DECAY=${13}
#    TAU=${14}
#    LR=${15}
#    TRACE_LENGTH=${18}  # is reset anyways
#    LAYER1=${19}
#    LAYER2=${20}
#    LAYER3=${21}
#    ACTIV_FCN=${22}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/DQN/$RUN_SCRIPT --instance_id $JOB_IDX --run_parallel True --runcount_limit $RUNCOUNT_LIMIT --logdir $LOGDIR --seed $JOB_IDX --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --eval_model $EVAL_MODEL --buffer_size $BUFFER_SIZE --update_interval $UPDATE_INTERVAL --batch_size $BATCH_SIZE"  #  --tau $TAU --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --gamma $GAMMA --lr $LR --trace_length $TRACE_LENGTH"
else
    echo "method "$MTHD" is not known."
fi

echo "Run python command in singularity container"
echo $PYCMD
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_local_p36.img echo "Hola"
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_local_p36.img $PYCMD
