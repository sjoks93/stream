from zigzag.workload.layer_node import LayerNode, LayerNodeAttributes
from zigzag.parser.workload_validator import WorkloadValidator
from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.mapping import InterCoreMappingAttributes
from stream.workload.computation.computation_node import ComputationNode
from stream.hardware.architecture.accelerator import Accelerator
from typing import Any
import re
from zigzag.workload.layer_attributes import (
    InputOperandSource,
    LayerDimRelation,
    LayerDimSizes,
    LayerEquation,
    LayerOperandPrecision,
    LayerPadding,
)
from zigzag.datatypes import (
    LayerDim,
    LayerOperand,
    UnrollFactor,
    Constants,
)

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
    ) -> DNNWorkloadStream:  # type: ignore
        self.accelerator = accelerator
        self.workload_data = workload_data
        self.all_mappings = all_mappings

    def create(self):
        
        node_list: list[LayerNode] = []
        for layer_data in self.workload_data:
            # TODO: don't create layer note but only extract the attributes
            cn_factory = ComputeNodeFactory(layer_data, self.all_mappings, self.accelerator)
            layer_node = cn_factory.create()
            node_list.append(layer_node)

        return DNNWorkloadStream(node_list)

class ComputeNodeFactory:
    """Creates a LayerNode instance from a validated and normalized user definition of a single workload layer"""

    def __init__(
        self, 
        node_data: dict[str, Any], 
        all_mappings: dict[str, InterCoreMappingAttributes],
        accelerator: Accelerator,
    ) -> None:
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
        node_attr = self.create_node_attr()
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
    
    def create_node_attr(self) -> LayerNodeAttributes:
        # From node data
        layer_type: str = self.node_data["operator_type"]
        equation = self.create_equation()
        layer_dim_sizes = self.create_layer_dim_sizes()
        operand_precision = self.create_operand_precision()
        dimension_relations = self.create_layer_dim_relations()
        constant_operands = self.create_constant_operands()
        [is_temporal, time_dim, is_state, state_operand] = self.create_temporal_operands()
        input_operand_source = self.create_operand_source()
        padding = self.create_padding()
        pr_layer_dim_sizes = self.create_pr_layer_dim_sizes()

        return LayerNodeAttributes(
            layer_type=layer_type,
            equation=equation,
            layer_dim_sizes=layer_dim_sizes,
            operand_precision=operand_precision,
            dimension_relations=dimension_relations,
            constant_operands=constant_operands,
            is_temporal=is_temporal,
            time_dim=time_dim,
            is_state=is_state,
            state_operand=state_operand,
            input_operand_source=input_operand_source,
            padding=padding,
            pr_layer_dim_sizes=pr_layer_dim_sizes,
        )
    
    def create_equation(self) -> LayerEquation:
        equation: str = self.node_data["equation"]
        equation = equation.replace("+=", "=")
        equation = equation.replace("++", "+")
        equation = equation.replace("*", " * ")
        equation = equation.replace("=", " = ")
        equation = equation.replace("+", " + ")
        return LayerEquation(equation)

    def create_layer_dim_sizes(self) -> LayerDimSizes:
        loop_dims = [LayerDim(x) for x in self.node_data["loop_dims"]]
        loop_sizes: list[UnrollFactor] = self.node_data["loop_sizes"]

        data = {dim: size for dim, size in zip(loop_dims, loop_sizes)}
        return LayerDimSizes(data)

    def create_operand_precision(self) -> LayerOperandPrecision:
        precisions: dict[str, int] = self.node_data["operand_precision"]
        data: dict[LayerOperand, int] = {LayerOperand(operand_str): size for operand_str, size in precisions.items()}
        return LayerOperandPrecision(data)

    def create_layer_dim_relations(self) -> list[LayerDimRelation]:
        relations: list[LayerDimRelation] = []
        for relation_str in self.node_data["dimension_relations"]:
            match = re.search(WorkloadValidator.LAYER_DIM_RELATION_REGEX, relation_str)
            assert match is not None
            dim_1, coef_2, dim_2, coef_3, dim_3 = match.groups()
            layer_dim_relation = LayerDimRelation(
                dim_1=LayerDim(dim_1),
                dim_2=LayerDim(dim_2),
                dim_3=LayerDim(dim_3),
                coef_2=int(coef_2) if coef_2 is not None else 1,
                coef_3=int(coef_3) if coef_3 is not None else 1,
            )
            relations.append(layer_dim_relation)

        return relations

    def create_temporal_operands(self):
        if self.node_data["operator_type"] == "SConv":
            is_state = True
            state_operand = Constants.STATE_LAYER_OP
        else:
            is_state = False
            state_operand = None
        if self.node_data["time_dim"] == "":
            if Constants.TIME_DIM.name in self.node_data["loop_dims"]:
                is_temporal = True
                time_dim = Constants.TIME_DIM
            else:
                is_temporal = False
                time_dim = None
        else:
            is_temporal = True
            time_dim = LayerDim(self.node_data["time_dim"])

        return [is_temporal, time_dim, is_state, state_operand]

    def create_constant_operands(self) -> list[LayerOperand]:
        operand_sources: dict[str, int] = self.node_data["operand_source"]
        constant_operands: list[str] = [op for op, source in operand_sources.items() if source == self.node_data["id"]]
        return [LayerOperand(layer_op_str) for layer_op_str in constant_operands]

    def create_operand_source(self) -> InputOperandSource:
        operand_sources: dict[str, int] = self.node_data["operand_source"]
        return {
            LayerOperand(layer_dim_str): source
            for layer_dim_str, source in operand_sources.items()
            if source != self.node_data["id"]
        }

    def create_padding(self) -> LayerPadding:
        if "pr_loop_dims" not in self.node_data or self.node_data["pr_loop_dims"] is None:
            return LayerPadding.empty()
        if "padding" not in self.node_data or self.node_data["padding"] is None:
            return LayerPadding.empty()

        pr_layer_dims: list[LayerDim] = [LayerDim(x) for x in self.node_data["pr_loop_dims"]]
        # length of the inner list equals 2
        padding_data: list[list[int]] = self.node_data["padding"]
        padding_dict: dict[LayerDim, tuple[int, int]] = {
            layer_dim: (padding_data[i][0], padding_data[i][1]) for i, layer_dim in enumerate(pr_layer_dims)
        }
        return LayerPadding(padding_dict)

    def create_pr_layer_dim_sizes(self) -> LayerDimSizes | None:
        if "pr_loop_dims" not in self.node_data or self.node_data["pr_loop_dims"] is None:
            return None
        if "pr_loop_sizes" not in self.node_data or self.node_data["pr_loop_sizes"] is None:
            return None

        pr_layer_dims: list[LayerDim] = [LayerDim(x) for x in self.node_data["pr_loop_dims"]]
        pr_sizes: list[UnrollFactor] = self.node_data["pr_loop_sizes"]
        size_dict = {layer_dim: size for layer_dim, size in zip(pr_layer_dims, pr_sizes)}
        return LayerDimSizes(size_dict)
   