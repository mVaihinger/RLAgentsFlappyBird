#!/bin/bash

while getopts r:m:l:e:d: opt
do
    case "${opt}" in
        r) NUM_AGENTS=${OPTARG};;
        m) MTHD=${OPTARG};;
        l) JOB_RESOURCES=${OPTARG};;
        e) EXP_TYPE=${OPTARG};;
        d) DIR_NAME=${OPTARG};;
#        v) VARS+=${OPTARG};;
    esac
done
echo "experiment: "$EXP_TYPE
echo "setup name: "$DIR_NAME
DATE=$(date +%Y_%m_%d_%H%M%S)
VARS=${@:11}

echo ${MTHD}" algorithm params: "${VARS}
echo "date: "$DATE

LOGDIR=/work/ws/nemo/fr_mv135-rlAgents-0/logs/experiments/$EXP_TYPE/$DIR_NAME/
mkdir -p $LOGDIR
echo "made logdir "$LOGDIR

MSUB_OUTPUT=$LOGDIR/msub_output
mkdir -p $MSUB_OUTPUT
echo "made msub output dir "$MSUB_OUTPUT

for ((idx_job=1; idx_job<=$NUM_AGENTS; idx_job++)); do
#for ((idx_job=17; idx_job<=($NUM_AGENTS+17); idx_job++)); do
    echo -e "Submitting agent: "${idx_job}
#    echo "$(date +%Y_%m_%d_%H%M%S)"

    ARGS=(${MTHD} ${LOGDIR} ${DATE} ${idx_job} ${VARS})
    echo ARGUMENTS
    echo "${ARGS[*]}"

    msub -N agent-${idx_job} -j oe -o $MSUB_OUTPUT/output.o${idx_job} -v SCRIPT_FLAGS="${ARGS[*]}" -l ${JOB_RESOURCES} /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/run_cont_agent.sh
done
