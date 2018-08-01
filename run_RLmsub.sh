#!/bin/bash

while getopts m:l: opt
do
    case "${opt}" in
        m) MTHD=${OPTARG};;
        l) JOB_RESOURCES=${OPTARG};;
    esac
done
DATE=$(date +%Y_%m_%d_%H%M%S)
VARS=${@:5}

echo ${MTHD}" algorithm params: "${VARS}
echo "date: "$DATE

LOGDIR=$HOME/logs/$MTHD/simpleTrain/$DATE
mkdir -p $LOGDIR
echo "made logdir "$LOGDIR

MSUB_OUTPUT=$LOGDIR/msub_output
mkdir -p $MSUB_OUTPUT
echo "made msub output dir "$MSUB_OUTPUT

ARGS=(${MTHD} ${LOGDIR} ${DATE} ${VARS})
echo ARGUMENTS
echo "${ARGS[*]}"

msub -N ${MTHD} -j oe -o $MSUB_OUTPUT/output.o -v SCRIPT_FLAGS="${ARGS[*]}" -l ${JOB_RESOURCES} /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/run_RL.sh
