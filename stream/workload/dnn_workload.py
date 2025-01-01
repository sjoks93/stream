import logging
from copy import deepcopy
from typing import Any


from stream.workload.node import Node
from stream.workload.workload_abc import WorkloadABC
logger = logging.getLogger(__name__)


class DNNWorkloadStream(WorkloadABC):
    def __init__(self, **attr: Any):
        """
        Collect all the algorithmic workload information here.
        Similar to `DNNWorkload` from ZigZag, but returns a DiGraph of ComputationNodes instead of LayerNodes.

        :return (self): Directed Graph with nodes the layers and edges the connections between layers.
        """
        super().__init__(**attr)  # type: ignore
        self.layer_id_to_obj: dict[int, Node] = {}

    def add(self, nodes: list[Node]):
        self.layer_node_list = nodes
        edges: list[tuple[Node, Node]] = []
        for node in nodes:
            node_name = f"{node.type}_{node.id}"
            # Add to graph
            logger.info("Parsed layer node %s", node_name)
            self.layer_id_to_obj[node.id] = node
            self.add_node(node)
            # Find all of its operand sources and add edges accordingly
            for parent_id in node.input_operand_source.values():
                # for parent_id in parent_list:
                if parent_id not in self.layer_id_to_obj:
                    raise ValueError(f"Illegal reference to non-existent layer with id {parent_id}")
                parent_node = self.layer_id_to_obj[parent_id]
                edges.append((parent_node, node))
        self.add_edges_from(edges)

    def get_copy_no_dummy(self) -> "DNNWorkloadStream":
        """Return a copy. DNNWorkloads don't contain DummyNodes in the first place."""
        return deepcopy(self)