import re
import numpy as np

from homology import HomologyGroup


def subtuples_of_tuple(vertex_tuple):
    stack = [(tuple(), vertex_tuple)]
    while stack:
        partial, remaining = stack.pop()
        if remaining:
            stack.extend((partial + remaining[:take], remaining[1:]) for take in [0, 1])
        else:
            yield partial


def subtuples_of_tuples(vertex_tuples):
    return set(subtuple for vertex_tuple in vertex_tuples for subtuple in subtuples_of_tuple(vertex_tuple))


def display_vector(vector, basis):
    if vector.shape != (len(basis),):
        raise ValueError(f"Expected a vector of length {len(basis)}")
    term_list = []
    for value, simplex in zip(vector, basis):
        if value == 0:
            continue
        elif value == 1 and not term_list:
            coefficient = ""
        elif value == 1:
            coefficient = " + "
        elif value == -1 and not term_list:
            coefficient = "-"
        elif value == -1:
            coefficient = " - "
        elif value > 0 and not term_list:
            coefficient = str(value)
        elif value > 0:
            coefficient = f" + {value}"
        elif not term_list:
            coefficient = f"-{abs(value)}"
        else:
            coefficient = f" - {abs(value)}"
        term_list.append(coefficient + str(simplex))
    if term_list:
        return ''.join(term_list)
    return "0"


def parse_vector(input_string, basis):
    basis_strings = [str(b).replace(' ', '') for b in basis]
    monomial_patterns = [re.compile(r"([\+|-]?)(\d*)" + re.escape(basis_string)) for basis_string in basis_strings]
    vector = np.zeros(len(basis), dtype=np.int64)
    search_string = input_string.replace(' ', '')
    for i, pattern in enumerate(monomial_patterns):
        for match in re.finditer(pattern, search_string):
            sign = {'': 1, '+': 1, '-': -1}[match.group(1)]
            coef = sign if not match.group(2) else sign * int(match.group(2))
            vector[i] += coef
    return vector


class SimplicialComplexPair:
    def __init__(self, facets, subcomplex_facets):
        sorted_facets = [tuple(sorted(facet)) for facet in facets]
        sorted_subcomplex_facets = [tuple(sorted(facet)) for facet in subcomplex_facets]
        self.face_set = subtuples_of_tuples(sorted_facets).union((tuple(),))
        for face in sorted_subcomplex_facets:
            if face not in self.face_set:
                raise ValueError(f"Subcomplex facet {face} is not present in the containing complex")

        self.subcomplex_face_set = subtuples_of_tuples(sorted_subcomplex_facets).union((tuple(),))
        self.subcomplex_faces = sorted(list(self.subcomplex_face_set))
        self.additional_faces = sorted(list(self.face_set.difference(self.subcomplex_face_set)))
        self.faces = self.additional_faces + self.subcomplex_faces
        self.k = len(self.additional_faces)
        self.face_lookup = {face: index for index, face in enumerate(self.additional_faces)}
        self._boundary_matrix = None
        self._homology = None

    def parse_chain(self, chain_string, truncate=False):
        search_string = chain_string.replace(' ', '')
        paren_matcher = re.compile(r"\(.*?\)")
        for match in re.finditer(paren_matcher, search_string):
            if eval(match.group(0)) not in self.faces:
                raise SyntaxError(
                    f"Unknown basis vector {match.group(0)}.  Are the vertices in increasing order? " +
                    "Are singleton tuples written (x,) as in python?  \n"
                )
        integer_combination_of_tuples_pattern = re.compile(r"0|[\+-]?\d*\(.*?\)(?:[\+-]\d*\(.*?\))*")
        if not re.match(integer_combination_of_tuples_pattern, search_string):
            raise SyntaxError(f"Error parsing input string. \n")
        vector = parse_vector(search_string, self.faces)
        if truncate:
            return vector[:self.k]
        return vector

    def chain_string(self, chain_vector, truncated=False):
        if truncated:
            return display_vector(chain_vector, self.additional_faces)
        return display_vector(chain_vector, self.faces)

    def boundary_matrix(self):
        if self._boundary_matrix is None:
            self._boundary_matrix = np.zeros((self.k, self.k), dtype=np.int64)
            for index, simplex in enumerate(self.additional_faces):
                if len(simplex) == 1:
                    continue
                for omit_index in range(len(simplex)):
                    face = simplex[:omit_index] + simplex[omit_index + 1:]
                    if face in self.subcomplex_faces:
                        continue
                    face_index = self.face_lookup[face]
                    self._boundary_matrix[index, face_index] = 1 if omit_index % 2 == 0 else -1
        return self._boundary_matrix

    def boundary(self, chain_string):
        vector = self.parse_chain(chain_string, truncate=True)
        return self.chain_string(vector @ self.boundary_matrix(), truncated=True)

    def homology(self):
        if self._homology is None:
            self._homology = HomologyGroup(self.boundary_matrix(), self.boundary_matrix())
        return self._homology

    def extend_matrix(self, truncated_cycles):
        zero_block = np.zeros((truncated_cycles.shape[0], len(self.subcomplex_faces)), dtype=np.int64)
        return np.hstack((truncated_cycles, zero_block))

    def truncate_matrix(self, cycles):
        return cycles[:, :-len(self.subcomplex_faces)]


class SimplicialComplex(SimplicialComplexPair):
    def __init__(self, facets):
        super().__init__(facets, [])


def write_homology_matrix(homology, output_cycles):
    matrix_rows = [np.zeros((0, homology.generating_cycles.shape[0]), dtype=np.int64)]
    for i in range(output_cycles.shape[0]):
        matrix_rows.append(homology.homology_vector(output_cycles[i]))
    return np.vstack(matrix_rows)


class LongExactSequence:
    def __init__(self, facets, subcomplex_facets):
        self.rel = SimplicialComplexPair(facets, subcomplex_facets)
        self.sub = SimplicialComplex(subcomplex_facets)
        self.all = SimplicialComplex(facets)

    def inclusion(self):
        cycles = self.sub.extend_matrix(self.sub.homology().generating_cycles)
        inclusion_indices = [self.all.faces.index(face) for face in self.sub.faces]
        inclusion = np.identity(len(self.all.faces), dtype=np.int64)[inclusion_indices]
        output_cycles = self.all.truncate_matrix(cycles @ inclusion)
        return write_homology_matrix(self.all.homology(), output_cycles)

    def projection(self):
        cycles = self.all.extend_matrix(self.all.homology().generating_cycles)
        projection_indices = [self.rel.faces.index(face) for face in self.all.faces]
        projection = np.identity(len(self.rel.faces), dtype=np.int64)[projection_indices]
        output_cycles = self.rel.truncate_matrix(cycles @ projection)
        return write_homology_matrix(self.rel.homology(), output_cycles)

    def connecting(self):
        cycles = self.rel.extend_matrix(self.rel.homology().generating_cycles)
        lift_indices = [self.all.faces.index(face) for face in self.rel.faces]
        lift = np.identity(len(self.all.faces), dtype=np.int64)[lift_indices]
        full_cycles = self.all.extend_matrix(self.all.truncate_matrix(cycles @ lift) @ self.all.boundary_matrix())
        inclusion_indices = [self.all.faces.index(face) for face in self.sub.faces]
        inclusion = np.identity(len(self.all.faces), dtype=np.int64)[inclusion_indices]
        output_cycles = self.sub.truncate_matrix(full_cycles @ np.transpose(inclusion))
        return write_homology_matrix(self.sub.homology(), output_cycles)

    def triple(self):
        return self.inclusion(), self.projection(), self.connecting()

    def pairs(self):
        return self.sub, self.all, self.rel
