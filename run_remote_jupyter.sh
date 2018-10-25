#!/bin/bash
# Set these values leave date empty if all runs shall be monitored in tensorboard
while getopts n: opt
do
    case "${opt}" in
        n) NODE=${OPTARG};;
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
ARGS=($WORKSPCDIR)

# -t option to kill all process running on the remote server when Ctrl-C in the local terminal to close the ssh connection.
# -L option binds the remote port $PORT to port 16006 on the localhost.
#ssh -t -L 16006:localhost:$PORT $USERHPC"@"$HOST "bash \$HOME/src/RLAgentsFlappyBird/run_jupyter.sh ${ARGS[@]}"
ssh -t -o "ProxyCommand ssh ${USERHPC}@login.nemo.uni-freiburg.de nc -w15 %h %p" -o "LocalForward localhost:1234 localhost:8888" ${USERHPC}"@"${NODE}.nemo.privat "bash \$HOME/src/RLAgentsFlappyBird/run_jupyter.sh ${ARGS[@]}"