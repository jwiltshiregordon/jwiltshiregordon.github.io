import traceback
from collections import Counter

from utils import remove_class
from js import document, MathJax

from simplicial_complex import SimplicialComplex, SimplicialComplexPair, LongExactSequence


def click_action(input_element_ids, output_element_id, func):
    inputs = [document.getElementById(eid) for eid in input_element_ids]
    output = document.getElementById(output_element_id)
    def action():
        try:
            result = func(*[el.value for el in inputs])
        except Exception as e:
            result = str(e) + "With traceback \n" + traceback.format_exc()
        output.innerHTML = result
        remove_class(Element(output_element_id), "hidden")
        MathJax.typeset()
    return action


def generate_all_faces(facets_string):
    tuples = eval(f"[{facets_string}]")
    simplicial_complex = SimplicialComplex(tuples)
    return f"all faces: <br/> {simplicial_complex.faces}"


p1 = click_action(["P1-input"], "P1-output", generate_all_faces)


def boundary_on_four_simplex(chain_string):
    simplicial_complex = SimplicialComplex([(1, 2, 3, 4, 5)])
    return simplicial_complex.boundary(chain_string)


p2 = click_action(["P2-input"], "P2-output", boundary_on_four_simplex)


def generating_cycles_for_pair(facets_string, subcomplex_facets_string):
    pair = SimplicialComplexPair(eval(f"[{facets_string}]"), eval(f"[{subcomplex_facets_string}]"))
    return '<br/>'.join(pair.chain_string(row, truncated=True) for row in pair.homology().cycles)


p3 = click_action(["P3-input-X", "P3-input-A"], "P3-output", generating_cycles_for_pair)



def generating_boundaries_for_pair(facets_string, subcomplex_facets_string):
    pair = SimplicialComplexPair(eval(f"[{facets_string}]"), eval(f"[{subcomplex_facets_string}]"))
    return '<br/>'.join(pair.chain_string(row @ pair.boundary_matrix(), truncated=True) for row in pair.homology().fillers)


p4 = click_action(["P4-input-X", "P4-input-A"], "P4-output", generating_boundaries_for_pair)


def check_if_cycle_and_witness_boundary(facets_string, subcomplex_facets_string, chain_string):
    pair = SimplicialComplexPair(eval(f"[{facets_string}]"), eval(f"[{subcomplex_facets_string}]"))
    chain_vector = pair.parse_chain(chain_string, truncate=True)
    if (chain_vector @ pair.boundary_matrix()).any():
        return "Not a cycle"
    cycle, error = pair.homology().standardize_cycle(chain_vector)
    if cycle.any():
        return "Cycle, but not a boundary"
    fillers_chain_string = pair.chain_string(error @ pair.homology().fillers, truncated=True)
    return f"Boundary of {fillers_chain_string}"


p5 = click_action(["P5-input-X", "P5-input-A", "P5-input"], "P5-output", check_if_cycle_and_witness_boundary)


def check_cycle_mobius(chain_string):
    pair = SimplicialComplexPair(
        [(1, 4, 5), (1, 2, 5), (2, 3, 6), (2, 5, 6), (1, 3, 4), (1, 3, 6)],
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
    )
    chain_vector = pair.parse_chain(chain_string, truncate=True)
    if (chain_vector @ pair.boundary_matrix()).any():
        return "Not a cycle"
    cycle, error = pair.homology().standardize_cycle(chain_vector)
    if cycle.any():
        return "Cycle, but not a boundary"
    fillers_chain_string = pair.chain_string(error @ pair.homology().fillers, truncated=True)
    return f"Boundary of {fillers_chain_string}"


p6 = click_action(["P6-input"], "P6-output", check_cycle_mobius)


def info_for_pair(facets_string, subcomplex_facets_string):
    pair = SimplicialComplexPair(eval(f"[{facets_string}]"), eval(f"[{subcomplex_facets_string}]"))
    h = pair.homology()
    fillers = '<br/>'.join([pair.chain_string(row, truncated=True) for row in h.fillers])
    cycles = '<br/>'.join([pair.chain_string(row, truncated=True) for row in h.cycles])
    divisors = str(h.elementary_divisors)
    return '<br/>'.join(["Standard fillers:", fillers, "Standard cycles:", cycles, "Divisor chain", divisors])


p7 = click_action(["P7-input-X", "P7-input-A"], "P7-output", info_for_pair)


def standardize_cycle(facets_string, subcomplex_facets_string, chain_string):
    pair = SimplicialComplexPair(eval(f"[{facets_string}]"), eval(f"[{subcomplex_facets_string}]"))
    h = pair.homology()
    fillers = '<br/>'.join([pair.chain_string(row, truncated=True) for row in h.fillers])
    generating_cycles = '<br/>'.join([pair.chain_string(row, truncated=True) for row in h.generating_cycles])
    chain_vector = pair.parse_chain(chain_string, truncate=True)
    if (chain_vector @ pair.boundary_matrix()).any():
        return "Not a cycle"
    cycle, error = h.standardize_cycle(chain_vector)
    return '<br/>'.join([
        "This cycle is homologous to a sum of the standard generators with coefficients",
        str(list(cycle[h.ones_count:])),
        "And the error term is the boundary of a sum of standard fillers with coefficients",
        str(list(error[:len(error) - h.zeros_count])),
        "<br/>For reference, here are the standard homology generators:",
        generating_cycles,
        "Standard fillers:",
        fillers,
    ])


p8 = click_action(["P8-input-X", "P8-input-A", "P8-input"], "P8-output", standardize_cycle)


mathjax_matrix_open = r"""
\begin{bmatrix}
"""

mathjax_matrix_close = r"""
\end{bmatrix}
"""


def mathjax_matrix(np_matrix):
    matrix_strings = [mathjax_matrix_open]
    for row in np_matrix:
        matrix_strings.append(' & '.join([str(entry) for entry in row]) + r"\\")
    matrix_strings.append(mathjax_matrix_close)
    return ''.join(matrix_strings)


def mathjax_cyclic_group(mod):
    integers = r"\mathbb{Z}"
    if mod == 0:
        return integers
    return fr"({integers} / {mod} )"


def mathjax_cyclic_group_with_multiplicity(mod, multiplicity):
    base_group = mathjax_cyclic_group(mod)
    if multiplicity == 1:
        return base_group
    return base_group + r"^{\oplus " + str(multiplicity) + r" }"


def mathjax_abelian_group(factors):
    if len(factors) == 0:
        return "0"
    summands = [mathjax_cyclic_group_with_multiplicity(*mm) for mm in Counter(factors).items()]
    return r" \oplus ".join(summands)


def les_for_pair(facets_string, subcomplex_facets_string):
    les = LongExactSequence(eval(f"[{facets_string}]"), eval(f"[{subcomplex_facets_string}]"))
    a, b, c = [mathjax_matrix(m) for m in les.triple()]
    ha, hx, hxa = [mathjax_abelian_group(p.homology().pruned_divisors) for p in les.pairs()]
    info_rows = [
        [r"H(A, \emptyset)", r"\longrightarrow", r"H(X, \emptyset)", r"\longrightarrow", r"H(X, A)", r"\longrightarrow", r"H(A, \emptyset)"],
        [ha, r"\longrightarrow", hx, r"\longrightarrow", hxa, r"\longrightarrow", ha],
        ["", a, "", b, "", c, ""],
    ]
    return r"\[ \begin{array} {rr} " + r"\\".join(" & ".join(row) for row in info_rows) + r"\\ \end{array} \]"


p9 = click_action(["P9-input-X", "P9-input-A"], "P9-output", les_for_pair)

p9_pairs = [
    [
        [(1, 2, 5), (2, 3, 5), (3, 4, 5), (1, 4, 5), (1, 2, 6), (2, 3, 6), (3, 4, 6), (1, 4, 6)],
        [(1, 2), (2, 3), (3, 4), (1, 4)],
    ],
    [
        [(1, 2), (2, 3), (3, 4), (3, 5), (1, 6), (6, 7), (1, 8), (4, 5), (2, 5), (7, 8)],
        [(1, 2), (2, 3), (3, 4), (3, 5), (1, 6), (6, 7), (1, 8)],
    ],
    [
        [
            (1, 2, 5), (1, 4, 5), (2, 3, 6), (2, 5, 6), (1, 3, 4), (3, 4, 6),
            (4, 5, 8), (4, 7, 8), (5, 6, 9), (5, 8, 9), (4, 6, 7), (6, 7, 9),
            (2, 7, 8), (1, 2, 7), (3, 8, 9), (2, 3, 8), (1, 7, 9), (1, 3, 9),
        ],
        [(1,)],
    ],
    [
        [
            (1, 3, 11), (2, 3, 7), (1, 9, 10), (4, 7, 11), (5, 7, 11), (1, 3, 7),
            (1, 6, 8), (1, 2, 6), (2, 4, 7), (2, 5, 11), (2, 4, 10), (1, 5, 7),
            (3, 4, 5), (2, 8, 9), (1, 2, 11), (2, 3, 5), (6, 9, 10), (1, 4, 9),
            (1, 4, 5), (3, 4, 11), (2, 6, 9), (1, 8, 10), (4, 8, 9), (4, 6, 10),
            (4, 6, 8), (2, 8, 10),
        ],
        [
            (1, 2), (1, 4), (2, 4),
        ]
    ],
    [
        [
            (1, 3, 11), (2, 3, 7), (1, 9, 10), (4, 7, 11), (5, 7, 11), (1, 3, 7),
            (1, 6, 8), (1, 2, 6), (2, 4, 7), (2, 5, 11), (2, 4, 10), (1, 5, 7),
            (3, 4, 5), (2, 8, 9), (1, 2, 11), (2, 3, 5), (6, 9, 10), (1, 4, 9),
            (1, 4, 5), (3, 4, 11), (2, 6, 9), (1, 8, 10), (4, 8, 9), (4, 6, 10),
            (4, 6, 8), (2, 8, 10),
        ],
        [
            (2, 7), (2, 11), (7, 11),
        ]
    ],
    [
        [
            (1, 2, 5), (1, 4, 5), (2, 3, 6), (2, 5, 6),
            (4, 5, 8), (4, 7, 8), (5, 6, 9), (5, 8, 9),
        ],
        [
            (1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9),
            (1, 4), (4, 7), (2, 5), (5, 8), (3, 6), (6, 9),
        ],
    ],
    [
        [
            (1, 2, 5), (1, 4, 5), (2, 3, 6), (2, 5, 6), (1, 3, 4), (3, 4, 6),
            (4, 5, 8), (4, 7, 8), (5, 6, 9), (5, 8, 9), (4, 6, 7), (6, 7, 9),
            (2, 7, 8), (1, 2, 7), (3, 8, 9), (2, 3, 8), (1, 7, 9), (1, 3, 9),
        ],
        [(1, 5), (5, 9), (1, 9), (2, 6), (4, 6), (4, 8), (2, 8)],
    ],
]

p9_input_facets = document.getElementById("P9-input-X")
p9_input_subcomplex_facets = document.getElementById("P9-input-A")


def p9_example_action(id_number):
    facets, subcomplex_facets = p9_pairs[id_number - 1]

    def action():
        p9_input_facets.value = str(facets)[1:-1]
        p9_input_subcomplex_facets.value = str(subcomplex_facets)[1:-1]

    return action


p9_1 = p9_example_action(1)
p9_2 = p9_example_action(2)
p9_3 = p9_example_action(3)
p9_4 = p9_example_action(4)
p9_5 = p9_example_action(5)
p9_6 = p9_example_action(6)
p9_7 = p9_example_action(7)
