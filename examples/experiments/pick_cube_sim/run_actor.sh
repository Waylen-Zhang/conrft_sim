export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \

# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export JAX_PLATFORM_NAME=cpu
# export TF_CPP_MIN_LOG_LEVEL=3
# export CUDA_VISIBLE_DEVICES=-1
# export CUDA_VISIBLE_DEVICES=1 \
export CUDA_HOME=/home/dx/miniconda3/envs/hilserl \

python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --actor \


# python ../../train_rlpd.py "$@" \
#     --exp_name=pick_cube_sim \
#     --checkpoint_path=first_run \
#     --actor \