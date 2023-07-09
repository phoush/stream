from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.core import Core
from stream.inputs.exploration.hardware.memory_pool.Stream_journal_exploration import *

def get_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=dram_single_port,
        operands=("I1", "I2", "O"),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
            {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
            {"fh": "rw_port_1", "tl": "rw_port_1", "fl": "rw_port_1", "th": "rw_port_1"},
        ),
        served_dimensions="all",
    )

    # from visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def get_operational_array():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = float("inf")
    multiplier_area = 0
    dimensions = {"D1": 1, "D2": 1}
    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_offchip_core(id):
    """This file defines an off-chip "core". Only the memory information of this core is important.
    The operational array is taken randomly.
    The user should make sure that none of the layers are actually mapped to this core.
    """
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    core = Core(id, operational_array, memory_hierarchy)
    return core


if __name__ == "__main__":
    print(get_offchip_core(0))