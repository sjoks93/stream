import logging as _logging
import re

from stream.api import optimize_allocation_co
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_path = "stream/inputs/examples/workload/resnet18.yaml"
mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
mode = "fused"
##############################################################################################

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = workload_path.split("/")[-1].split(".")
if wl_name[1] == "onnx":
    wl_type = "ONNX"
    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
elif wl_name[1] == "yaml":
    wl_type = "YAML"
    layer_stacks = [tuple(range(0, 7)), tuple(range(7, 14))] + list((i,) for i in range(14, 31))

else:
    raise f"Invalid workload name {wl_name[0]}.{wl_name[1]}"

experiment_id = f"{hw_name}-{wl_name[0]}-{mode}-{wl_name[1]}-constraint_optimization-test"
######################################################################

scme = optimize_allocation_co(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    stack_types=None,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=True,
    wl_type=wl_type,
)

############PLOTTING#############
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#################################

#########################PLOTTING PATHS##############################
timeline_fig_path_plotly = f"outputs/{experiment_id}/schedule.html"
memory_fig_path = f"outputs/{experiment_id}/memory.png"
json_path = f"outputs/{experiment_id}/scme.json"
#####################################################################

#####################CostModelEvaluationLUT LOAD#############################
cost_lut_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)
#############################################################################

# Plotting schedule timeline of best SCME
visualize_timeline_plotly(
    scme,
    draw_dependencies=draw_dependencies,
    draw_communication=plot_data_transfer,
    fig_path=timeline_fig_path_plotly,
    cost_lut=cost_lut,
)
# Plotting memory usage of best SCME
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)

# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
