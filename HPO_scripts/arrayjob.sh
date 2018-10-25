#!/bin/bash
if [ -n "${SCRIPT_FLAGS}" ] ; then
        if [ -z "${*}" ]; then
                set -- ${SCRIPT_FLAGS}
        fi
fi
MTHD=${1}
LOGDIR=${2}
DATE=${3}

cd $HOME/arrayjob

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "Job id:                               $MOAB_JOBID"
echo "Job name:                             $MOAB_JOBNAME"
echo "Number of nodes allocated to job:     $MOAB_NODECOUNT"
echo "Number of cores allocated to job:     $MOAB_PROCCOUNT"

# TODO add bohb flags
# how to set instance_id and array_id??

if [ "$MTHD" == "A2C" ] || [ "$MTHD" == "LSTM_A2C" ] || [ "$MTHD" == "GRU_A2C" ]; then
    RUN_SCRIPT=a2c_bohb_wrapper.py

    # VARS
    ARCHITECTURE=${4}
    ENV=${5}
    TEST_ENV=${6}
    MIN_STEPS=${7}
    MAX_STEPS=${8}
    MAX_GRAD_NORM=${9}
    LOG_INTERVAL=${10}
    EVAL_MODEL=${11}

    NENVS=${12}
    BATCH_SIZE=${13}

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

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/A2C/$RUN_SCRIPT --instance_id $MOAB_JOBID --array_id $MOAB_JOBARRAYINDEX --logdir $LOGDIR --seed $MOAB_JOBARRAYINDEX --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --max_resource $MAX_STEPS --min_resource $MIN_STEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --eval_model $EVAL_MODEL --nenvs $NENVS --batch_size $BATCH_SIZE"   # --gamma $GAMMA --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --nsteps $NSTEPS --lr $LR"
elif [ "$MTHD" == "DQN" ]; then
    RUN_SCRIPT=dqn_bohb_wrapper.py

    ARCHITECTURE=${4}
    ENV=${5}
    TEST_ENV=${6}
    MIN_STEPS=${7}
    MAX_STEPS=${8}
    MAX_GRAD_NORM=${9}
    LOG_INTERVAL=${10}
    EVAL_MODEL=${11}

    BUFFER_SIZE=${12}
    UPDATE_INTERVAL=${13}

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

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/DQN/$RUN_SCRIPT --instance_id $MOAB_JOBID --array_id $MOAB_JOBARRAYINDEX --logdir $LOGDIR --seed $MOAB_JOBARRAYINDEX --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --max_resource $MAX_STEPS --min_resource $MIN_STEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --eval_model $EVAL_MODEL --buffer_size $BUFFER_SIZE --update_interval $UPDATE_INTERVAL"  #  --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --gamma $GAMMA --lr $LR"
elif [ "$MTHD" == "LSTM_DQN" ] || [ "$MTHD" == "GRU_DQN" ]; then
    RUN_SCRIPT=dqn_bohb_wrapper.py

    ARCHITECTURE=${4}
    ENV=${5}
    TEST_ENV=${6}
    MIN_STEPS=${7}
    MAX_STEPS=${8}
    MAX_GRAD_NORM=${9}
    LOG_INTERVAL=${10}
    EVAL_MODEL=${11}

    BUFFER_SIZE=${12}
    UPDATE_INTERVAL=${13}
    BATCH_SIZE=${14}

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

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/DQN/$RUN_SCRIPT --instance_id $JOB_IDX --array_id $MOAB_JOBARRAYINDEX --logdir $LOGDIR --seed $JOB_IDX --architecture $ARCHITECTURE --env $ENV --test_env $TEST_ENV --max_resource $MAX_STEPS --min_resource $MIN_STEPS --max_grad_norm $MAX_GRAD_NORM --log_interval $LOG_INTERVAL --eval_model $EVAL_MODEL --buffer_size $BUFFER_SIZE --update_interval $UPDATE_INTERVAL --batch_size $BATCH_SIZE"  #  --tau $TAU --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --gamma $GAMMA --lr $LR --trace_length $TRACE_LENGTH"
else
    echo "method "$MTHD" is not known."
fi

# PYCMD="python3.6 script.py $MOAB_JOBARRAYINDEX"

module load tools/singularity/2.5
echo "Run python command in singularity container"
echo $PYCMD
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_bohb_p36.img $PYCMD
