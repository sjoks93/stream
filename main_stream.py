from zigzag.classes.stages import *
from stream.classes.stages import *
import re

# Initialize the logger
import logging as _logging
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

#################################
accelerator = 'stream.inputs.examples.hardware.Meta_prototype_dual_core_simd_offchip'
# workload_path = 'stream.inputs.examples.workload.resnet18'
workload_path = 'stream/inputs/examples/workload/resnet18.onnx'
mapping_path = 'stream.inputs.examples.mapping.meta_prototype_quad_core_pooling_simd_offchip'

CN_define_mode = 1  # manually define outer CN size for all cores and all layers
hint_loops = [('OY', 'all')]  # outer CN loops, with error in resnet18 plotting

hw_name = accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", workload_path)[-1]
experiment_id = f"{hw_name}-{wl_name}-CNmode_{CN_define_mode}-hintloop_{str(hint_loops)}"
node_hw_cost_pkl_name = f'saved_CN_HW_cost-{experiment_id}'
plot_file_name = f'-{experiment_id}-'
plot_full_schedule = True
plot_data_transfer = True
#################################


mainstage = MainStage([  # Initializes the MainStage as entry point
    AcceleratorParserStage,  # Parses the accelerator
    StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
    # UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
    GenerateCNWorkloadHybridStage,
    IntraCoreMappingStage,
    InterCoreMappingStage,
],

    accelerator=accelerator,  # required by AcceleratorParserStage
    workload_path=workload_path,  # required by ModelParserStage
    mapping_path=mapping_path,  # required by ModelParserStage
    loma_lpf_limit=6,  # required by LomaStage
    nb_ga_individuals=4,  # number of individuals in each genetic algorithm generation
    nb_ga_generations=1,  # number of genetic algorithm generations
    node_hw_performances_path=f"outputs/{node_hw_cost_pkl_name}.pickle",  # saved node_hw_performances to skip re-computation
    plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
    plot_file_name=plot_file_name,
    plot_full_schedule=plot_full_schedule,
    plot_data_transfer=plot_data_transfer,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    scheduler_candidate_selection='memory'
)

# Launch the MainStage
mainstage.run()