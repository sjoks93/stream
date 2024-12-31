import logging
from copy import deepcopy
from typing import Any

from zigzag.utils import DiGraphWrapper

from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
logger = logging.getLogger(__name__)


class DNNWorkloadStream(ComputationNodeWorkload):
    def __init__(self, nodes: list[ComputationNode], **attr: Any):
        """
        Collect all the algorithmic workload information here.
        Similar to `DNNWorkload` from ZigZag, but returns a DiGraph of ComputationNodes instead of LayerNodes.

        :return (self): Directed Graph with nodes the layers and edges the connections between layers.
        """
        super().__init__(**attr)  # type: ignore

        layer_id_to_obj: dict[int, ComputationNode] = {}
        self.layer_node_list = nodes

        for node in nodes:
            node_name = f"{node.type}_{node.id}"



            # Add to graph
            logger.info("Parsed layer node %s", node_name)
            layer_id_to_obj[node.id] = node
            self.add_node(node)

            # Find all of its operand sources and add edges accordingly
            edges: list[tuple[ComputationNode, ComputationNode]] = []
            for _, parent_id in node.input_operand_source.items():
                # for parent_id in parent_list:
                if parent_id not in layer_id_to_obj:
                    raise ValueError(f"Illegal reference to non-existent layer with id {parent_id}")
                parent_node = layer_id_to_obj[parent_id]
                edges.append((parent_node, node))
            self.add_edges_from(edges)

    def get_copy_no_dummy(self) -> "DNNWorkloadStream":
        """Return a copy. DNNWorkloads don't contain DummyNodes in the first place."""
        return deepcopy(self)