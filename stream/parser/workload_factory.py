from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.mapping import InterCoreMappingAttributes
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.node import Node
from stream.hardware.architecture.accelerator import Accelerator
from zigzag.parser.workload_factory import LayerNodeFactory
from typing import Any


class WorkloadFactoryStream():
    """Generates a `Workload` instance from the validated and normalized user-provided data.
    Almost identical to ZigZagWorkloadFactory, apart from the return type: DNNWorkloadStream instead of
    DNNWorkload
    """

    def __init__(
        self,
        workload_data: list[dict[str, Any]],
        all_mappings: dict[str, InterCoreMappingAttributes],
        accelerator: Accelerator,
    ):
        self.accelerator = accelerator
        self.workload_data = workload_data
        self.all_mappings = all_mappings

    def create(self) -> DNNWorkloadStream:  # type: ignore
        node_list: list[Node] = []
        DNN_workload = DNNWorkloadStream()
        for layer_data in self.workload_data:
            # TODO: don't create layer note but only extract the attributes
            cn_factory = ComputeNodeFactory(layer_data, self.all_mappings, self.accelerator)
            layer_node = cn_factory.create()
            node_list.append(layer_node)
        DNN_workload.add(node_list)
        return DNN_workload

class ComputeNodeFactory:
    """Creates a LayerNode instance from a validated and normalized user definition of a single workload layer"""

    def __init__(
        self, 
        node_data: dict[str, Any], 
        all_mappings: dict[str, InterCoreMappingAttributes],
        accelerator: Accelerator,
    ):
        """!
        @node_data validated and normalized user-defined data for a single workload layer
        @mapping_data validated and normalized user-defined data for all mappings, or None is case no mapping-related
        instances need to be constructed
        """
        self.node_data = node_data
        self.all_mappings = all_mappings
        self.node_id: int = self.node_data["id"]
        self.node_name: str = self.node_data["name"] if self.node_data["name"] is not None else f"Layer{self.node_id}"
        self.accelerator = accelerator

    def create(self) -> ComputationNode:
        node_factory = LayerNodeFactory(self.node_data, mapping_data=[])
        node_attr = node_factory.create_node_attr()        
        mapping_attr = self.create_mapping_attr()
        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node_name,
            node_attr=node_attr,
            mapping_attr=mapping_attr,
            op_type = node_attr.layer_type
        )

    def create_mapping_attr(self):
        """Get the mapping that corresponds to this node's operator. Replace the spatial mapping with the corresponding
        core's dataflows.
        NOTE The core's dataflow always precedes the mapping's spatial mapping
        TODO Mapping based on node name instead of note operator is not yet supported
        """
        default_mapping = self.all_mappings["default"]
        if self.node_name in self.all_mappings:
            mapping = self.all_mappings[self.node_name]
        elif self.node_data["operator_type"] in self.all_mappings:
            mapping = self.all_mappings[self.node_data["operator_type"]]
        else:
            mapping = default_mapping

        # Override spatial mapping by the one defined in the core's dataflows
        try:
            core_dataflow = self.accelerator.get_spatial_mapping_from_core(mapping.core_allocation)
            mapping.spatial_mapping = core_dataflow
        except ValueError:
            pass
        # If no inter/intra mapping is given: use default one
        if not mapping.intra_core_tiling:
            mapping.intra_core_tiling = default_mapping.intra_core_tiling
        if not mapping.inter_core_tiling:
            mapping.inter_core_tiling = default_mapping.inter_core_tiling
        return mapping
    