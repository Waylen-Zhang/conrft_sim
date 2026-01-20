export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=flexiv_assembly \
    --checkpoint_path=/home/dx/waylen/conrft/examples/experiments/pick_cube_sim/conrft2 \
    --actor \
    # --eval_checkpoint_step=26000 \