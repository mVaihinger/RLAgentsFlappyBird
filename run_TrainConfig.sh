#!/bin/bash

while getopts r:m:l:o: opt
do
    case "${opt}" in
        r) NUM_AGENTS=${OPTARG};;
        m) MTHD=${OPTARG};;
        l) JOB_RESOURCES=${OPTARG};;
        o) HO_TYPE=${OPTARG};;
#        v) VARS+=${OPTARG};;
    esac
done
DATE=$(date +%Y_%m_%d_%H%M%S)
VARS=${@:9}

echo ${MTHD}" algorithm params: "${VARS}
echo "date: "$DATE

LOGDIR=$HOME/logs/test_configs/$HO_TYPE/$MTHD/$DATE
mkdir -p $LOGDIR
echo "made logdir "$LOGDIR

MSUB_OUTPUT=$LOGDIR/msub_output
mkdir -p $MSUB_OUTPUT
echo "made msub output dir "$MSUB_OUTPUT

for ((idx_job=1; idx_job<=$NUM_AGENTS; idx_job++)); do
    echo -e "Submitting agent: "${idx_job}
#    echo "$(date +%Y_%m_%d_%H%M%S)"

    ARGS=(${MTHD} ${LOGDIR} ${DATE} ${idx_job} ${VARS})
    echo ARGUMENTS
    echo "${ARGS[*]}"

    msub -N agent-${idx_job} -j oe -o $MSUB_OUTPUT/output.o${idx_job} -v SCRIPT_FLAGS="${ARGS[*]}" -l ${JOB_RESOURCES} /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/run_cont_agent.sh
done
