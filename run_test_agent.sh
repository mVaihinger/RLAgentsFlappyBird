#!/bin/bash
if [ -n "${SCRIPT_FLAGS}" ] ; then
	if [ -z "${*}" ]; then
		set -- ${SCRIPT_FLAGS}
	fi
fi
MTHD=${1}
LOGDIR=${2}
SEED=${3}
#DATE=${4}

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "Job id:                               $MOAB_JOBID"
echo "Job name:                             $MOAB_JOBNAME"
echo "Number of nodes allocated to job:     $MOAB_NODECOUNT"
echo "Number of cores allocated to job:     $MOAB_PROCCOUNT"

echo -e "\nchanging directory"
cd /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/  # $MTHD/
module load tools/singularity/2.6

# TODO add argument regarding the mthd to pycommand
# same run_script for all methods.

TEST_ENV=${4}
TOTAL_TESTSTEPS=${5}
LOG_INTERVAL=${6}
RESULTFILE=${7}
#SEED=${8}

PYCMD="python3.6 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/testing.py --method $MTHD --logdir $LOGDIR --test_env $TEST_ENV --total_timesteps $TOTAL_TESTSTEPS --result_file $RESULTFILE"

echo "Run python command in singularity container"
echo $PYCMD
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_experiments_p36.img echo "Hola"
singularity exec /work/ws/nemo/fr_mv135-rlAgents-0/containers/rlAgents_experiments_p36.img $PYCMD
