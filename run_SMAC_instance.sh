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
    RUN_SCRIPT=a2c_smac_wrapper.py

    # VARS
    RUNCOUNT_LIMIT=${5}
    ENV=${6}
    POLICY=${7}
    TOTAL_TIMESTEPS=${8}
    MAX_GRAD_NORM=${9}
    NENVS=${10}
    GAMMA=${11}
    LOG_INTERVAL=${12}
    SHOW_INTERVAL=${13}
    TEST_INTERVAL=${14}

    NSTEPS=${15}
    LR=${16}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/A2C/$RUN_SCRIPT --instance_id $JOB_IDX --run_parallel True --runcount_limit $RUNCOUNT_LIMIT --logdir $LOGDIR --seed $JOB_IDX --env $ENV --policy $POLICY --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --nenvs $NENVS --gamma $GAMMA --log_interval $LOG_INTERVAL --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --nsteps $NSTEPS --lr $LR"
#elif [ "$MTHD" == "RA2C" ]; then
#    RUN_SCRIPT=a2c_smac_wrapper.py
#
#    # VARS
#    RUNCOUNT_LIMIT=${5}
#    ENV=${6}
#    POLICY=${7}
#    TOTAL_TIMESTEPS=${8}
#    MAX_GRAD_NORM=${9}
#    NENVS=${10}
#    GAMMA=${11}
#    LOG_INTERVAL=${12}
#    SHOW_INTERVAL=${13}
#
#    NSTEPS=${14}
#    LR=${15}
#
#    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/$RUN_SCRIPT --instance_id $JOB_IDX --run_parallel True --runcount_limit $RUNCOUNT_LIMIT --logdir $LOGDIR --seed $JOB_IDX --env $ENV --policy $POLICY --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --nenvs $NENVS --gamma $GAMMA --log_interval $LOG_INTERVAL"
elif [ "$MTHD" == "DQN" ]; then
    RUN_SCRIPT=dqn_smac_wrapper.py

    # VARS
    RUNCOUNT_LIMIT=${5}
    ENV=${6}
    TOTAL_TIMESTEPS=${7}
    MAX_GRAD_NORM=${8}
    BUFFER_SIZE=${9}
    LOG_INTERVAL=${10}
    SHOW_INTERVAL=${11}
    TEST_INTERVAL=${12}

    GAMMA=${13}
    LR=${14}
    BATCH_SIZE=${15}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/$RUN_SCRIPT --instance_id $JOB_IDX --run_parallel True --runcount_limit $RUNCOUNT_LIMIT --logdir $LOGDIR --seed $JOB_IDX --env $ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --buffer_size $BUFFER_SIZE --log_interval $LOG_INTERVAL --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --gamma $GAMMA --lr $LR --batch_size $BATCH_SIZE"
elif [ "$MTHD" == "DRQN" ]; then
    RUN_SCRIPT=drqn_smac_wrapper.py

    # VARS
    RUNCOUNT_LIMIT=${5}
    ENV=${6}
    TOTAL_TIMESTEPS=${7}
    TAU=${8}
    MAX_GRAD_NORM=${9}
    BUFFER_SIZE=${10}
    UPDATE_INTERVAL=${11}
    LOG_INTERVAL=${12}
    SHOW_INTERVAL=${13}
    TEST_INTERVAL=${14}

    GAMMA=${15}
    LR=${16}
    TRACE_LENGTH=${17}
    NBATCH=${18}

    PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/$RUN_SCRIPT --instance_id $JOB_IDX --run_parallel True --runcount_limit $RUNCOUNT_LIMIT --logdir $LOGDIR --seed $JOB_IDX --env $ENV --total_timesteps $TOTAL_TIMESTEPS --tau $TAU --max_grad_norm $MAX_GRAD_NORM --buffer_size $BUFFER_SIZE --update_interval $UPDATE_INTERVAL --log_interval $LOG_INTERVAL --show_interval $SHOW_INTERVAL --test_interval $TEST_INTERVAL --gamma $GAMMA --lr $LR --trace_length $TRACE_LENGTH --nbatch $NBATCH"
else  # TODO add DQN_RNN and A2C_RNN
    echo "method "$MTHD" is not known."
fi

echo "Run python command in singularity container"
echo $PYCMD
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_local_p36.img echo "Hola"
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_local_p36.img $PYCMD
