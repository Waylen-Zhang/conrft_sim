from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig
from experiments.pick_cube_sim.config import TrainConfig as PickCubeTrainConfig
from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
# from experiments.flexiv_assembly.config import TrainConfig as FlexivAssemblyConfig

CONFIG_MAPPING = {
    "task1_pick_banana": PickBananaTrainConfig,
    "pick_cube_sim": PickCubeTrainConfig,
    "ram_insertion": RAMInsertionTrainConfig,
    # "flexiv_assembly": FlexivAssemblyConfig,
}
