from abc import ABCMeta
from zigzag.utils import DiGraphWrapper
from stream.workload.node import Node

class WorkloadABC(DiGraphWrapper[Node], metaclass=ABCMeta):
    def get_copy_no_dummy(self) -> "WorkloadABC": ...

