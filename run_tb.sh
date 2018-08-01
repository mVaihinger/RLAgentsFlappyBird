#!/bin/bash

LOGDIR=${1}
#LOGDIR=$HOME/logs/$DIR
#LOGDIR=/home/fr/fr_fr/fr_mv135/logs/A2C/2018_07_19_083354/a2c_output1/

WORKSPCDIR=${2}
PORT=${3}

module load tools/singularity/2.5
#singularity exec "${WORKSPCDIR}"/rlAgents.img echo ${LOGDIR}
singularity exec "${WORKSPCDIR}"/containers/rlAgents_smac_pip3.img tensorboard --logdir=${LOGDIR} --host localhost --port ${PORT}