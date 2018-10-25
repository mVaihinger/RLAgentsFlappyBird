#!/bin/bash

#LOGDIR=${1}
#LOGDIR=$HOME/logs/$DIR
#LOGDIR=/home/fr/fr_fr/fr_mv135/logs/A2C/2018_07_19_083354/a2c_output1/

WORKSPCDIR=${1}
#PORT=${3}

module load tools/singularity/2.6
#singularity exec "${WORKSPCDIR}"/rlAgents.img echo ${LOGDIR}
#singularity exec "${WORKSPCDIR}"/containers/rlAgents_smac_pip3.img tensorboard --logdir=${LOGDIR} --host localhost --port ${PORT}

CMD="jupyter notebook --no-browser --port 8888 --notebook-dir="${WORKSPCDIR}
singularity exec -B "${WORKSPCDIR}"/logs/notebooks:/run/user ${WORKSPCDIR}/containers/rlAgents_experiments_jupp36.img $CMD
