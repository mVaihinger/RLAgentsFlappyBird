#!/bin/bash

while getopts r:m:l:e:d: opt
do
    case "${opt}" in
        r) NUM_AGENTS=${OPTARG};;
        m) MTHD=${OPTARG};;
        l) JOB_RESOURCES=${OPTARG};;
        e) EXP_TYPE=${OPTARG};;
        d) DIR_NAME=${OPTARG};;
    esac
done
DATE=$(date +%Y_%m_%d_%H%M%S)
VARS=${@:11}

echo ${MTHD}" algorithm params: "${VARS}
echo "date: "$DATE

LOGDIR=/work/ws/nemo/fr_mv135-rlAgents-0/logs/experiments/$EXP_TYPE/$DIR_NAME
mkdir -p $LOGDIR
echo "logdir "$LOGDIR

if [ "$MTHD" == "A2C" ] || [ "$MTHD" == "LSTM_A2C" ] || [ "$MTHD" == "GRU_A2C" ]; then
    OUTPUTDIR=a2c_output
elif [ "$MTHD" == "DQN" ] || [ "$MTHD" == "LSTM_DQN" ] || [ "$MTHD" == "GRU_DQN" ]; then
    OUTPUTDIR=dqn_output
elif [ "$MTHD" == "PPO" ] || [ "$MTHD" == "LSTM_PPO" ] || [ "$MTHD" == "GRU_PPO" ]; then
    OUTPUTDIR=ppo_output
fi

MSUB_OUTPUT=$LOGDIR/msub_output
mkdir -p $MSUB_OUTPUT
echo "msub output dir "$MSUB_OUTPUT

for ((idx_job=1; idx_job<=20; idx_job++)); do
#    echo -e "Submitting agent: "${idx_job}
    # only one job per experiment setup.
    O_DIR=$LOGDIR/${OUTPUTDIR}${idx_job}
    echo $O_DIR

    ARGS=(${MTHD} ${O_DIR} 42 ${VARS})
    echo ARGUMENTS
    echo "${ARGS[*]}"

    msub -N agent -j oe -o $MSUB_OUTPUT/track_output.o${idx_job} -v SCRIPT_FLAGS="${ARGS[*]}" -l ${JOB_RESOURCES} /home/fr/fr_fr/fr_mv135/src/RLAgentsFlappyBird/run_tracking_agent.sh
done