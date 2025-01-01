import logging
from typing import Any

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class SchedulingOrderGenerationStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        workload: ComputationNodeWorkload,
        **kwargs: dict[str, Any],
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        self.layer_stacks = kwargs.get("layer_stacks", None)  # optional
        self.stack_types = kwargs.get("stack_types", None)
        self.scheduling_order = None

    def run(self):
        if self.layer_stacks:
            # All nodes of earlier stacks should be scheduled before later stacks
            self.scheduling_order = []
            for layer_stack in self.layer_stacks:
                nodes = [n for n in self.workload.nodes() if n.id in layer_stack]
                self.scheduling_order.extend(sorted(((n.id, n.sub_id) for n in nodes), reverse=True))
        else:
            # Generate a list of node ids from highest priority to lowest
            # We give higher priority to nodes deeper in the graph
            self.scheduling_order = sorted(((n.id, n.sub_id) for n in self.workload.nodes()), reverse=True)

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["workload"] = self.workload
        self.kwargs["scheduling_order"] = self.scheduling_order
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

