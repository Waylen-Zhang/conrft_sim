export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.35 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
export XLA_FLAGS="--xla_gpu_autotune_level=0" && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=flexiv_assembly \
    --checkpoint_path=/home/dx/waylen/conrft/examples/experiments/flexiv_assembly/conrft \
    --actor \
    # --eval_checkpoint_step=26000 \