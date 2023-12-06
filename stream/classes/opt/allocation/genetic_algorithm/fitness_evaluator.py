from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.opt.scheduling.layer_stacks import get_layer_stacks, LayerStackMode
from stream.classes.opt.splitting.cn_graph import get_cn_graph
from stream.classes.workload.computation_node import ComputationNode
from zigzag.utils import pickle_deepcopy
import networkx as nx
import numpy as np

from copy import deepcopy

from stream.utils import get_too_large_operands


class FitnessEvaluator:
    def __init__(
        self, workload=None, accelerator=None, node_hw_performances=None
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.node_hw_performances = node_hw_performances
        # self.num_cores = len(inputs.accelerator.cores)

    def get_fitness(self):
        raise NotImplementedError

class CoreBasedFitnessEvaluator(FitnessEvaluator):
    """The core based fitness evaluator considers latency cost."""

    def __init__(
        self,
        workload,
        accelerator,
        core_based_accelerator
        #original_workload,  # used for layer stack calculation
    ) -> None:
        super().__init__(workload, accelerator)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]
        self.accelerator = accelerator
        self.core_based_accelerator = core_based_accelerator
        self.workload = workload
        self.constant_operand_occupation_factor = 1
        self.layer_stacks_mode = LayerStackMode.CORE_BASED
        self.scheduler_candidate_selection = 'latency'

    def get_fitness(self, core_allocations: list, return_scme=False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        layer_stacks, workload_with_allocations = get_layer_stacks(
            workload=self.workload,
            accelerator=self.accelerator,
            mode=self.layer_stacks_mode,
            core_allocations = core_allocations
        )
        core_allocations = self.update_core_based_accelerator(layer_stacks, 
                workload_with_allocations, core_allocations)
        # generate new graph here
        w = deepcopy(workload_with_allocations)
        G = get_cn_graph(w, self.core_based_accelerator)
        self.cn_graph = G
        self.set_cost_cn_nodes()
        
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.cn_graph),
            pickle_deepcopy(self.core_based_accelerator),
            self.scheduler_candidate_selection,
            [],
            layer_stacks,
        )
        scme.run()
        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme


    def update_core_based_accelerator(self, layer_stacks, workload_with_allocations, core_allocations):
        updated_core_allocations = []
        core_nodes = []
        i = 0
        for stack in layer_stacks:
            for stack_core in stack:
                core_0 = next((x for x in self.accelerator.cores.nodes() if x.id == core_allocations[i]),None)
                layer_node = next((x for x in workload_with_allocations.nodes() if isinstance(x, ComputationNode) and x.id[0] == stack_core),None)
                core_tmp = deepcopy(core_0)
                core_tmp.id = i
                layer_node.mappings[core_tmp] = layer_node.mappings[core_0]

                core_nodes.append(core_tmp)
                updated_core_allocations.append(i)
                layer_node.core_allocation = i
                layer_node.attrs['core_allocation'] = i
                layer_node.set_core_allocation(i)
                i += 1
        G = nx.Graph()
        for ii_n, nn in enumerate(core_nodes[1:]):
            G.add_edge(core_nodes[ii_n],nn)
        self.core_based_accelerator.cores = deepcopy(G)
        return updated_core_allocations

    def set_cost_cn_nodes(self):
        for node in nx.topological_sort(self.cn_graph):
            latency = np.prod([x[1] for x in node.producer_inner_loops])

            node.set_onchip_energy(1)
            node.set_offchip_energy(1)
            node.set_runtime(latency)
            node.set_too_large_operands(False)


class StandardFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally."""

    def __init__(
        self,
        workload,
        accelerator,
        node_hw_performances,
        layer_groups_flexible,
        scheduler_candidate_selection,
        operands_to_prefetch,
        original_workload,  # used for layer stack calculation
    ) -> None:
        super().__init__(workload, accelerator, node_hw_performances)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.layer_groups_flexible = layer_groups_flexible
        self.scheduler_candidate_selection = scheduler_candidate_selection
        self.operands_to_prefetch = operands_to_prefetch
        self.original_workload = original_workload
        self.constant_operand_occupation_factor = 1
        self.layer_stacks_mode = LayerStackMode.OCCUPATION_BASED

    def get_fitness(self, core_allocations: list, return_scme=False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        layer_stacks = get_layer_stacks(
            self.workload,
            self.original_workload,
            self.accelerator,
            self.constant_operand_occupation_factor,
            self.layer_stacks_mode,
        )
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.scheduler_candidate_selection,
            self.operands_to_prefetch,
            layer_stacks,
        )
        scme.run()
        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

    def set_node_core_allocations(self, core_allocations):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            (layer_id, group_id) = self.layer_groups_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (
                node
                for node in self.workload.nodes()
                if isinstance(node, ComputationNode)
                and node.id[0] == layer_id
                and node.group == group_id
            )
            for node in nodes:
                try:
                    equivalent_unique_node = next(
                        (n for n in self.node_hw_performances.keys() if node == n)
                    )
                except StopIteration:
                    raise ValueError(
                        f"The given node_hw_performances doesn't have run information for node={node}"
                    )
                try:
                    cme = self.node_hw_performances[equivalent_unique_node][core]
                except KeyError:
                    raise KeyError(
                        f"The given node_hw_performances doesn't have information for core_allocation={core_allocation} of node={node}"
                    )
                onchip_energy = (
                    cme.energy_total
                )  # Initialize on-chip energy as total energy
                latency = cme.latency_total1
                too_large_operands = get_too_large_operands(
                    cme, self.accelerator, core_id=core_allocation
                )
                # If there is a too_large_operand, we separate the off-chip energy.
                offchip_energy = 0
                for too_large_operand in too_large_operands:
                    layer_operand = next(
                        (
                            k
                            for (k, v) in cme.layer.memory_operand_links.items()
                            if v == too_large_operand
                        )
                    )
                    layer_operand_offchip_energy = cme.energy_breakdown[layer_operand][
                        -1
                    ]
                    offchip_energy += layer_operand_offchip_energy
                    onchip_energy -= layer_operand_offchip_energy
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(latency)
                node.set_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)



