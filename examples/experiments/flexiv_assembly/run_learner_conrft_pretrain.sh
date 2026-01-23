export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
export XLA_FLAGS="--xla_gpu_autotune_level=0" && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=flexiv_assembly \
    --checkpoint_path=/home/dx/waylen/conrft/examples/experiments/flexiv_assembly/conrft \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=/home/dx/waylen/conrft/examples/experiments/flexiv_assembly/demo_data/flexiv_assembly_40_demos_2026-01-22_12-09-41.pkl \
    --pretrain_steps=19500 \
    --debug=False \
    --learner \
