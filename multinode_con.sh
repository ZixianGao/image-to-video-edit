# !/bin/bash

# Ensure execution is in the correct directory
cd $PBS_O_WORKDIR

# Activate environment (optional if already done in main script, but safe)
# export HOME=/work/gj40/w42010
source $HOME/anaconda3/bin/activate diffusion

# --- Distributed Training Setup ---
# Get MASTER_ADDR
export MASTER_ADDR=${MASTER_ADDR:-$(head -n 1 $PBS_NODEFILE)}
if [ -z "$MASTER_ADDR" ]; then
    export MASTER_ADDR=$(hostname)
fi



sort $PBS_NODEFILE | uniq -c | awk '{print $2 " slots=" $1}' > hostfile



# Set a fixed MASTER_PORT
export MASTER_PORT=34159 # You can choose any free port

# Determine WORLD_SIZE (total number of processes/nodes)
if [ -f "$PBS_NODEFILE" ]; then
    export WORLD_SIZE=$(wc -l < "$PBS_NODEFILE")
else
    export WORLD_SIZE=8 # Fallback for single node testing? Adjust if needed.
fi

# Determine NODE_RANK
export NODE_RANK=0
if [ -f "$PBS_NODEFILE" ]; then
    NODELIST=($(cat "$PBS_NODEFILE"))
    CURRENT_HOST=$(hostname)
    for i in "${!NODELIST[@]}"; do
        if [ "${NODELIST[$i]}" == "$CURRENT_HOST" ]; then
            export NODE_RANK=$i
            break
        fi
    done
fi
NUM_NODES=4
echo "Node Rank: $NODE_RANK, World Size: $WORLD_SIZE, Master Addr: $MASTER_ADDR, Master Port: $MASTER_PORT"



export CUDA_HOME=/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/compilers
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DS_BUILD_OPS=0
#------ Launch training ------#
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=1 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
  train_w_qwen.py \
    --num_nodes=$NUM_NODES