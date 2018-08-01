#!/bin/bash
# sync RLAgentsFlappyBird repository to nemo cluster
while getopts "u:h:" opt; do
    case "${opt}" in
    u)
        USERHPC=$OPTARG;;
    h)
        HOST=$OPTARG;;
    \?)
        echo "Invalid option: -$OPTARG"
        exit 1;;
    esac
done
CODE_DIR=/media/mara/OS/Users/Mara/Documents/Masterthesis/RLAgents

echo "Transferring: "$CODE_DIR"/RLAgentsFlappyBird"
ssh $USERHPC"@"$HOST "mkdir -p \$HOME/src/" < /dev/null
rsync -e"ssh -i ~/.ssh/id_rsa_NEMO.pub" -auv $CODE_DIR/RLAgentsFlappyBird $USERHPC"@"$HOST:"\$HOME/src/" < /dev/null

# sync container
#echo -e "\nTransferring: "$CODE_DIR"/containers"
#ssh $USERHPC"@"$HOST "mkdir -p \$HOME/containers" < /dev/null
##rsync -e"ssh -i ~/.ssh/id_rsa_NEMO.pub" -auv $CODE_DIR/containers/rlAgents_smac.img $USERHPC"@"$HOST:"/work/ws/nemo/fr_mv135-rlAgents-0/containers/" < /dev/null
##rsync -e"ssh -i ~/.ssh/id_rsa_NEMO.pub" -auv $CODE_DIR/containers/rlAgents_smac_tk.img $USERHPC"@"$HOST:"/work/ws/nemo/fr_mv135-rlAgents-0/containers/" < /dev/null
#rsync -e"ssh -i ~/.ssh/id_rsa_NEMO.pub" -auv $CODE_DIR/containers/rlAgents_local_p36.img $USERHPC"@"$HOST:"/work/ws/nemo/fr_mv135-rlAgents-0/containers/" < /dev/null
