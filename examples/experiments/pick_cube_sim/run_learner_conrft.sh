export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.65 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
export XLA_FLAGS="--xla_gpu_autotune_level=0" && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=/home/dx/waylen/conrft/examples/experiments/pick_cube_sim/conrft \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=/home/dx/waylen/conrft/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_25_demos_2026-01-19_09-44-52.pkl \
    --pretrain_steps=10000 \
    --debug=False \
    --learner \