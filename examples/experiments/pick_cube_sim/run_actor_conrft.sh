export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.35 && \
export XLA_FLAGS="--xla_gpu_autotune_level=0" && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=/home/dx/waylen/conrft/examples/experiments/pick_cube_sim/conrft \
    --actor \
    # --eval_checkpoint_step=26000 \