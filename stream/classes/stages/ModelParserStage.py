from typing import Generator

from stream.classes.io.onnx.model import ONNXModelParser
from zigzag.classes.stages.Stage import Stage
from zigzag.utils import pickle_deepcopy
from stream.classes.workload.computation_node import ComputationNode
import networkx as nx

import logging

logger = logging.getLogger(__name__)

def sanitize_workload_graph(workload):
    # remove any node in the workload graph
    # that is not a computation node.
    # Reconnect edges from removed nodes
    for node in nx.topological_sort(workload):
        if not isinstance(node, ComputationNode):
            workload.add_edge(*workload.predecessors(node), *workload.successors(node))
            workload.remove_node(node)
    return workload


class ONNXModelParserStage(Stage):
    def __init__(
        self, list_of_callables, *, workload_path, mapping_path, accelerator, **kwargs
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.mapping_path = mapping_path
        self.onnx_model_parser = ONNXModelParser(
            workload_path, mapping_path, accelerator
        )

    def run(self) -> Generator:
        self.onnx_model_parser.run()
        onnx_model = self.onnx_model_parser.get_onnx_model()
        workload = self.onnx_model_parser.get_workload()
        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["mapping_path"] = self.mapping_path
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            onnx_model=onnx_model,
            workload=workload,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    # # For testing purposes
    # def is_leaf(self) -> bool:
    #     return True


def parse_workload_from_path(workload_path, mapping_path, accelerator):
    """
    Parse the input workload residing in accelerator_path.
    The "workload" dict is converted to a NetworkX graph.
    """
    import importlib
    from stream.classes.workload.dnn_workload import DNNWorkload

    # If workload_path is a string, then it is a path to a workload file
    if isinstance(workload_path, str):
        module = importlib.import_module(workload_path)
        workload = module.workload
    elif isinstance(workload_path, dict):
        # Make a copy here to prevent it being changed later
        workload = pickle_deepcopy(workload_path)
    else:
        raise NotImplementedError(f"Provided workload format ({type(workload_path)}) is not supported.")

    module = importlib.import_module(mapping_path)
    mapping = module.mapping

    workload = DNNWorkload(workload, mapping, accelerator)
    logger.info(
        f"Created workload graph with {workload.number_of_nodes()} nodes and {workload.number_of_edges()} edges."
    )

    return workload


class UserDefinedModelParserStage(Stage):
    def __init__(
        self, list_of_callables, *, workload_path, mapping_path, accelerator, **kwargs
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload_path = workload_path
        self.mapping_path = mapping_path
        self.accelerator = accelerator

    def run(self) -> Generator:
        workload = parse_workload_from_path(
            self.workload_path, self.mapping_path, self.accelerator
        )

        self.kwargs["accelerator"] = self.accelerator
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:], workload=workload, **self.kwargs
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
