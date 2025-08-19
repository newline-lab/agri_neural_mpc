#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_multi.sh [n_train] [trees_number]
# Examples:
#   ./run_multi.sh          # n_train=1, trees=25 (default)
#   ./run_multi.sh 3        # n_train=3, trees=25
#   ./run_multi.sh 3 100    # n_train=3, trees=100 -> ./unity_exec_100/neural_mpc_unity.x86_64

# Base ROS master and TCP port
uri=11311
port=10000

# tmux session base name
sessname="nmpc_multi_train#"

# Windows to create
PANE=(ROS_LAUNCH UNITY)

# Args
n_train=${1:-1}
trees=${2:-25}

# Resolve Unity executable path based on trees number
UNITY_DIR="./unity_exec_${trees}"
UNITY_EXEC="${UNITY_DIR}/neural_mpc_unity.x86_64"

if [[ ! -x "${UNITY_EXEC}" ]]; then
  echo "ERROR: Unity executable not found or not executable at: ${UNITY_EXEC}"
  echo "       Expected path pattern: ./unity_exec_${trees}/neural_mpc_unity.x86_64"
  exit 1
fi

for (( c=0; c<=n_train; c++ )); do
    new_port=$((port + c))
    new_uri=$((uri + c))

    sleep 1
    tmux new-session -d -s "${sessname}${c}"

    for ((t=0; t<${#PANE[@]}; t++)); do
        if (( t > 0 )); then
            tmux new-window -t "${sessname}${c}:${t}"
        fi
        tmux rename-window -t "${sessname}${c}:${t}" "${PANE[t]}"
        tmux send-keys -t "${sessname}${c}:${t}" "export ROS_MASTER_URI=http://127.0.0.1:${new_uri}" C-m
    done

    # Launch nmpc_ros baseline.launch with tcp_port
    tmux send-keys -t "${sessname}${c}:0" \
      "source ros/pyenv/bin/activate && source ros/devel/setup.bash && roslaunch -p ${new_uri} nmpc_ros baseline.launch tcp_port:=${new_port}" C-m

    # Launch Unity app with ros-ip and ros-port
    tmux send-keys -t "${sessname}${c}:1" \
      "\"${UNITY_EXEC}\" --ros-ip=127.0.0.1 --ros-port=${new_port}" C-m

    sleep 5
done
