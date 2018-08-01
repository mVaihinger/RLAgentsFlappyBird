#!/bin/bash
# Set these values leave date empty if all runs shall be monitored in tensorboard
while getopts d: opt
do
    case "${opt}" in
        d) LOGDIR=${OPTARG};;
#        m) MTHD=${OPTARG};;
    esac
done

# Start tb
USERHPC="fr_mv135"
HOST=login.nemo.uni-freiburg.de

# Sync data
chmod +x sync_code.sh
./sync_code.sh -u $USERHPC -h $HOST

WORKSPCDIR=/work/ws/nemo/fr_mv135-rlAgents-0
PORT=7777
ARGS=($LOGDIR $WORKSPCDIR $PORT)

# -t option to kill all process running on the remote server when Ctrl-C in the local terminal to close the ssh connection.
# -L option binds the remote port $PORT to port 16006 on the localhost.
ssh -t -L 16006:localhost:$PORT $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_tb.sh ${ARGS[@]}"