#!/bin/bash

while getopts w:m:l: opt
do
    case "${opt}" in
        w) NUM_WORKERS=${OPTARG};;
        m) MTHD=${OPTARG};;
        l) JOB_RESOURCES=${OPTARG};;
#        v) VARS+=${OPTARG};;
    esac
done
DATE=$(date +%Y_%m_%d_%H%M%S)
VARS=${@:7}

echo ${MTHD}" algorithm params: "${VARS}
echo "date: "$DATE

LOGDIR=/work/ws/nemo/fr_mv135-rlAgents-0/logs/BOHB/$MTHD/$DATE
#LOGDIR=$HOME/logs/HO/$MTHD/$DATE
mkdir -p $LOGDIR
echo "made logdir "$LOGDIR

MSUB_OUTPUT=$LOGDIR/msub_output
mkdir -p $MSUB_OUTPUT
echo "made msub output dir "$MSUB_OUTPUT

#for ((idx_job=1; idx_job<=$RUN_JOBS; idx_job++)); do
#    echo -e "Submitting job: "${idx_job}
#    echo "$(date +%Y_%m_%d_%H%M%S)"

ARGS=(${MTHD} ${LOGDIR} ${DATE} ${VARS})
echo ARGUMENTS
echo "${ARGS[*]}"

msub -t 1-$NUM_WORKERS -N bohb-worker -j oe -o $MSUB_OUTPUT/%J.o%I -v SCRIPT_FLAGS="${ARGS[*]}" -l ${JOB_RESOURCES} /home/fr/fr_fr/fr_mv135/arrayjob/arrayjob.moab
#    msub -N smac-run-${idx_job} -j oe -o $MSUB_OUTPUT/output.o${idx_job} -v SCRIPT_FLAGS="${ARGS[*]}" -l nodes=1:ppn=10,pmem=1gb,walltime=15:00:00 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/run_SMAC_instance.sh
#    msub -N smac-run-${idx_job} -j oe -o $MSUB_OUTPUT/output.o${idx_job} -v SCRIPT_FLAGS="${ARGS[*]}" -l nodes=1:ppn=10,pmem=1gb,walltime=1:00:00 /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/run_SMAC_instance.sh
#done
