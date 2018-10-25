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
cd /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/
module load tools/singularity/2.5

if [ "$MTHD" == "A2C" ] || [ "$MTHD" == "RA2C" ]; then
    RUN_SCRIPT=run_a2c.py

    # VARS
    ENV=${5}
    POLICY=${6}
    TOTAL_TIMESTEPS=${7}
    MAX_GRAD_NORM=${8}
    NENVS=${9}
    GAMMA=${10}
    LOG_INTERVAL=${11}
    TEST_INTERVAL=${12}
    SHOW_INTERVAL=${13}
    NSTEPS=${14}
    VF_COEFF=${15}
    ENT_COEFF=${16}
    LR=${17}
    UNITS_SHARED_HLAYER1=${18}  # (shared1, shared2, policy)
    UNITS_SHARED_HLAYER2=${19}
    UNITS_POLICY_LAYER=${20}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/A2C/$RUN_SCRIPT --keep_model 0 --logdir $LOGDIR --seed $JOB_IDX --env $ENV --policy $POLICY --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --nenvs $NENVS --gamma $GAMMA --log_interval $LOG_INTERVAL --test_interval $TEST_INTERVAL --nsteps $NSTEPS --vf_coeff $VF_COEFF --ent_coeff $ENT_COEFF --lr $LR --units_shared_layer1 $UNITS_SHARED_HLAYER1 --units_shared_layer2 $UNITS_SHARED_HLAYER2 --units_policy_layer $UNITS_POLICY_LAYER"
elif [ "$MTHD" == "DQN" ]; then
    RUN_SCRIPT=run_dqn.py

    # VARS
    ENV=${5}
    TOTAL_TIMESTEPS=${6}
    MAX_GRAD_NORM=${7}
    BUFFER_SIZE=${8}
    LOG_INTERVAL=${9}
    TEST_INTERVAL=${10}

    EPSILON=${11}
    EPS_DECAY=${12}
    BATCH_SIZE=${13}
    LR=${14}
    UNITS_LAYER1=${15}  # (shared1, shared2, policy)
    UNITS_LAYER2=${16}
    UNITS_LAYER3=${17}
    GAMMA=${18}

    UPDATE_INTERVAL=${19}
    SHOW_INTERVAL=${20}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/$RUN_SCRIPT --keep_model 0 --logdir $LOGDIR --seed $JOB_IDX --env $ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --buffer_size $BUFFER_SIZE --log_interval $LOG_INTERVAL --test_interval $TEST_INTERVAL --epsilon $EPSILON --epsilon_decay $EPS_DECAY --lr $LR --units_layer1 $UNITS_LAYER1 --units_layer2 $UNITS_LAYER2 --units_layer3 $UNITS_LAYER3 --gamma $GAMMA --update_interval $UPDATE_INTERVAL --show_interval $SHOW_INTERVAL"
elif [ "$MTHD" == "DRQN" ]; then
    RUN_SCRIPT=run_drqn.py

    # VARS
    ENV=${5}
    TOTAL_TIMESTEPS=${6}
    TAU=${7}
    MAX_GRAD_NORM=${8}
    BUFFER_SIZE=${9}
    UPDATE_INTERVAL=${10}
    LOG_INTERVAL=${11}
    TEST_INTERVAL=${12}

    EPSILON=${13}
    EPS_DECAY=${14}
    NBATCH=${15}
    TRACE_LENGTH=${16}
    LR=${17}
    UNITS_LAYER1=${18}  # (shared1, shared2, policy)
    UNITS_LAYER2=${19}
    UNITS_LAYER3=${20}
    GAMMA=${21}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/$RUN_SCRIPT --keep_model 0 --logdir $LOGDIR --seed $JOB_IDX --env $ENV --total_timesteps $TOTAL_TIMESTEPS --tau $TAU --max_grad_norm $MAX_GRAD_NORM --buffer_size $BUFFER_SIZE --update_interval $UPDATE_INTERVAL --log_interval $LOG_INTERVAL --test_interval $TEST_INTERVAL --epsilon $EPSILON --epsilon_decay $EPS_DECAY --nbatch $NBATCH --trace_length $TRACE_LENGTH --lr $LR --units_layer1 $UNITS_LAYER1 --units_layer2 $UNITS_LAYER2 --units_layer3 $UNITS_LAYER3 --gamma $GAMMA"
else  # TODO add DQN_RNN and A2C_RNN
    echo "method "$MTHD" is not known."
fi

echo "Run python command in singularity container"
echo $PYCMD
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_local_p36.img echo "Hola"
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_local_p36.img $PYCMD
