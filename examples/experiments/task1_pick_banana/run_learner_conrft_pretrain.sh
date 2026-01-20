export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=/root/online_rl/conrft/examples/experiments/pick_cube_sim/conrft\
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=./demo_data/task1_pick_banana_30_demos.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \
