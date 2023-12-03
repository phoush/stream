import itertools
from math import ceil, prod
from re import L
from typing import List, Dict
import networkx as nx
import numpy as np
from rtree import index
from zigzag.utils import pickle_deepcopy
from stream.classes.workload.elementwise_node import ElementwiseNode
from stream.classes.workload.flatten_node import FlattenNode
from stream.classes.workload.lpnormalization_node import LpNormalizationNode
from stream.classes.workload.reshape_node import ReshapeNode
from stream.classes.workload.transpose_node import TransposeNode
from stream.classes.workload.tensor import Tensor
from zigzag.classes.mapping.temporal.temporal_loop import TemporalLoop
from stream.classes.workload.computation_node import ComputationNode
from zigzag.classes.stages.Stage import Stage, MainStage
from zigzag.classes.stages import *
from stream.classes.workload.dummy_node import DummyNode
from stream.classes.opt.splitting.splitting import (
    convert_inner_cn_loops,
    convert_outer_cn_loops,
    convert_outer_cn_loops_with_k,
)

import logging

logger = logging.getLogger(__name__)


class GenerateNHPStage(Stage):
    """
    Class that transforms the layer-by-layer workload into finer CN workload graph.
    """

    def __init__(
        self,
        list_of_callables,
        *,
        workload,
        accelerator,
        cn_define_mode,
        hint_loops,
        **kwargs,
    ):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.loma_lpf_limit = 6
        self.loma_show_progress_bar = False
        self.node_hw_performances = {}

    def run(self):
        unique_finer_nodes = []
        # For each node get all the finer nodes and set the intra edges
        G = nx.DiGraph()
        unique_layers = set()
        unique_nodes = set()

        # Create set of all unique layers that have same loop dim size
        # so as to avoid temporal mapping, spatial mapping generation
        # of repeated layers
        for node in nx.topological_sort(self.workload):
            if not isinstance(
                node, ComputationNode
            ):  # If other node types shouldn't be included in finer node graph, add here
                continue
            print(node)
            print(node.loop_dim_size)
            if frozenset(node.loop_dim_size.items()) not in unique_layers:
                unique_layers.add(frozenset(node.loop_dim_size.items()))
                unique_nodes.add(tuple([frozenset(node.loop_dim_size.items()), node]))
            # HERE!
            # GENERATE DIFFERENT TM/SM PER CORE COMBINATION
        for _, node in unique_nodes:
            self.node_hw_performances[node] = {}
            for core in [c for c in self.accelerator.cores.nodes()]:
#                core = self.accelerator.get_core(core_id)
                node.core_allocation = core.id  # Set the node's core allocation to the core_id we want to extract hw performance for

                node.user_spatial_mapping_hint = (
                    {"D1": ["K", "OX"], "D2": ["C", "FX", "FY"], "D3":["K","C","OX","OY","G"]}
                )  # Set the node's spatial mapping to the possible spatial mappings of the current core
                # Initialize the flow that will be followed to extract the optimal HW performance of every unique node-core allocation
                main_stage = self.get_intra_core_mapping_flow(
                    # TODO too_large_operands assigned to hardcoded value, to be fixed later
                    too_large_operands=False,
                    node=node,
                    core_id=core.id,
                )
                answers = main_stage.run()
                assert (
                    len(answers) == 1
                ), "GenerateNHP MappingStage's subflow returned more than one CME"
                cme = answers[0][0]
                node.core_allocation = None  # Reset the node's core allocation
                self.node_hw_performances[node][core] = cme
        #        self.save_node_hw_performances()  # Save the hw performances dict after every node is finished

                pass
        for node in nx.topological_sort(self.workload):
            if not isinstance(
                node, ComputationNode
            ):  # If other node types shouldn't be included in finer node graph, add here
                continue
            node_key = frozenset(node.loop_dim_size.items())
            for n in self.node_hw_performances.keys():
                if node_key == frozenset(n.loop_dim_size.items()):
                    node.mappings = self.node_hw_performances[n]

 
        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["accelerator"] = self.accelerator
#        kwargs["node_hw_performances"] = self.node_hw_performances
        breakpoint()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

        yield None, None


    def get_intra_core_mapping_flow(self, node, too_large_operands, core_id):
        logger.info(
            f"Launching intra-core mapping optimization for {node} -> core {core_id} ..."
        )

        if too_large_operands:
            accelerator = self.add_offchip_to_core(
                core_id, too_large_operands, node.id[0]
            )
        else:
            accelerator = self.accelerator

        main_stage = MainStage(
            [  # Initializes the MainStage as entry point
                MinimalLatencyStage,
                SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
                MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
                LomaStage,  # Generates multiple temporal mappings (TM)
                CostModelStage,  # Evaluates generated SM and TM through cost model
            ],
            layer=node,
            accelerator=accelerator,  # required by a number of stages
            loma_lpf_limit=self.loma_lpf_limit,  # required by LomaStage
            loma_show_progress_bar=self.loma_show_progress_bar,
        )
        return main_stage


