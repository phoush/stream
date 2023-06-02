workload = {
    0: {
        "operator_type": "layer_on_core_0",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": ["ix=1*ox+1*fx", "iy=1*oy+1*fy"],
        "loop_dim_size": {
            "B": 1,
            "K": 64,
            "C": 256,
            "OY": 56,
            "OX": 56,
            "FY": 1,
            "FX": 1,
        },
        "pr_loop_dim_size": {"IY": 56, "IX": 56},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["I", "W"],
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "padding": {"IY": (0, 0), "IX": (0, 0)},
    },
    1: {
        "operator_type": "layer_on_core_1",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": ["ix=1*ox+1*fx", "iy=1*oy+1*fy"],
        "loop_dim_size": {
            "B": 1,
            "K": 64,
            "C": 64,
            "OY": 56,
            "OX": 56,
            "FY": 3,
            "FX": 3,
        },
        "pr_loop_dim_size": {"IY": 56, "IX": 56},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [0]},
        "constant_operands": ["W"],
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "padding": {"IY": (1, 1), "IX": (1, 1)},
    },
    2: {
        "operator_type": "layer_on_core_2",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": ["ix=1*ox+1*fx", "iy=1*oy+1*fy"],
        "loop_dim_size": {
            "B": 1,
            "K": 256,
            "C": 64,
            "OY": 56,
            "OX": 56,
            "FY": 1,
            "FX": 1,
        },
        "pr_loop_dim_size": {"IY": 56, "IX": 56},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [1]},
        "constant_operands": ["W"],
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "padding": {"IY": (0, 0), "IX": (0, 0)},
    },
    3: {
        "operator_type": "layer_on_core_3",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": ["ix=1*ox+1*fx", "iy=1*oy+1*fy"],
        "loop_dim_size": {
            "B": 1,
            "K": 64,
            "C": 256,
            "OY": 56,
            "OX": 56,
            "FY": 1,
            "FX": 1,
        },
        "pr_loop_dim_size": {"IY": 56, "IX": 56},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [2]},
        "constant_operands": ["W"],
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "padding": {"IY": (0, 0), "IX": (0, 0)},
    },
    4: {
        "operator_type": "layer_on_core_4",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": ["ix=1*ox+1*fx", "iy=1*oy+1*fy"],
        "loop_dim_size": {
            "B": 1,
            "K": 64,
            "C": 64,
            "OY": 56,
            "OX": 56,
            "FY": 3,
            "FX": 3,
        },
        "pr_loop_dim_size": {"IY": 56, "IX": 56},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [3]},
        "constant_operands": ["W"],
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "padding": {"IY": (1, 1), "IX": (1, 1)},
    },
}
