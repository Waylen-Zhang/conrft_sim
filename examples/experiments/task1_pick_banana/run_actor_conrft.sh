export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

python ../../train_conrft_octo.py "$@" \
    --exp_name=task1_pick_banana \
    --checkpoint_path=/root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft \
    --actor \
    # --eval_checkpoint_step=26000 \