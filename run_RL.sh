#!/bin/bash
if [ -n "${SCRIPT_FLAGS}" ] ; then
	if [ -z "${*}" ]; then
		set -- ${SCRIPT_FLAGS}
	fi
fi
MTHD=${1}
LOGDIR=${2}
DATE=${3}

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "Job id:                               $MOAB_JOBID"
echo "Job name:                             $MOAB_JOBNAME"
echo "Number of nodes allocated to job:     $MOAB_NODECOUNT"
echo "Number of cores allocated to job:     $MOAB_PROCCOUNT"

echo -e "\nchanging directory"
cd /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/${MTHD}/
module load tools/singularity/2.5

if [ "$MTHD" == "A2C" ]; then
    RUN_SCRIPT=run_a2c.py
    PYCMD="python3 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/$RUN_SCRIPT --logdir $LOGDIR --seed 300 --env "${4}" --total_timesteps "${5}" --max_grad_norm "${6}" --nenvs "${7}" --gamma "${8}" --log_interval "${9}" --nsteps "${10}
elif [ "$MTHD" == "DQN" ]; then
    ENV=${4}
    TOTAL_TIMESTEPS=${5}
    MAX_GRAD_NORM=${6}
    BUFFER_SIZE=${7}
    ACTIV_FCN=${8}
    LOG_INTERVAL=${9}
    TEST_ENV=${10}
    RUN_SCRIPT=run_dqn.py
    PYCMD="python3 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/$MTHD/$RUN_SCRIPT --logdir $LOGDIR --seed 1 --env $ENV --test_env $TEST_ENV --total_timesteps $TOTAL_TIMESTEPS --max_grad_norm $MAX_GRAD_NORM --buffer_size $BUFFER_SIZE --activ_fcn $ACTIV_FCN --log_interval $LOG_INTERVAL"
else
    echo "method "${MTHD}" is not known."
fi

echo "Run python command in singularity container"
echo ${PYCMD}
#singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_smac_tk.img echo "Hola"
#singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_smac_tk.img ${PYCMD}
#singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_smac_pip3.img echo "Hola"
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_smac_pip3.img ${PYCMD}