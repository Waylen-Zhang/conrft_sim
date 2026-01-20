export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
# export CUDA_HOME=/home/dx/miniconda3/envs/hilserl \

export CUDA_HOME=/home/dx/miniconda3/envs/hilserl \

python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=first_run \
    --demo_path="/home/dx/waylen/hilserl-sim/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_20_demos_2026-01-07_11-24-14.pkl" \
    --learner \