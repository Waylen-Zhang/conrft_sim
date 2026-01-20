export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=flexiv_assembly \
    --checkpoint_path=/home/dx/waylen/conrft/examples/experiments/flexiv_assembly/conrft \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=/home/dx/waylen/conrft/examples/experiments/flexiv_assembly/demo_data/pick_cube_sim_5_demos_2026-01-18_13-17-23.pkl \
    --pretrain_steps=2000 \
    --debug=False \
    --learner \