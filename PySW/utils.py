
"""
Title: Utils for PySW package
Date: 17 October 2024
Authors:
- Giovanni Francesco Diotallevi
- Irving Leander Reascos Valencia

DOI: doi.doi.doi

Description:
------------
This module contains various utility functions that support the PySW package. These utilities include functions 
for manipulating symbolic quantum operators, boson counting, creating multiplicative groups, and simplifying expressions.

Functions:
----------
1. `get_order`: Determines the order of a given expression or symbolic term.
2. `group_by_order`: Groups terms in an expression by their order.
3. `get_count_boson`: Returns the count of bosonic operators for a given operator.
4. `count_bosons`: Counts bosons in a given structure.
5. `count_bosonic_subspaces`: Determines the number of bosonic subspaces in a Hamiltonian.
6. `domain_expansion`: Expands a term over the given domains (finite, infinite).
7. `apply_commutation_relations`: Applies commutation relations to simplify expressions.
8. `separate_diagonal_off_diagonal`: Separates diagonal and off-diagonal terms in an expression.
9. `memoized`: A memoization decorator for caching function results.
10. `partitions`: Generates integer partitions.
11. `commutator`: Computes the commutator of two operators.
12. `display_dict`: Displays a dictionary of LaTeX expressions.
13. `group_by_operators`: Groups terms in an expression by quantum operators.
14. `get_perturbative_expression`: Constructs a perturbative expansion of a given expression.
15. `get_matrix`: Generates matrix representations for quantum operators.
16. `get_boson_matrix`: Generates matrix representations for bosonic operators.

Dependencies:
-------------
ipython==8.28.0
multimethod==1.12
numpy==2.1.2
sympy==1.13.3
"""


# Standard library imports
from itertools import product
from typing import Union

# Third-party imports
from IPython.display import display, Math
from multimethod import multimethod
from numpy import (any as np_any, all as np_all, array as np_array,
                   ndarray as np_ndarray, logical_not as np_logical_not,
                   ones as np_ones, vectorize as np_vectorize,
                   zeros as np_zeros)
from sympy import (Expr, Mul, Pow, Symbol, eye, kronecker_product, latex,
                   zeros as sp_zeros, sqrt as sp_sqrt, diag as sp_diag)
from sympy.core.numbers import (Float, Half, ImaginaryUnit, Integer, One, Rational)
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

# Local application/library imports
from .classes import MulGroup, RDSymbol, RDOperator, Expression

'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                        GROUP BY ORDER
    TO-DOS:
        [ ] This can be parallelized over the terms.
---------------------------------------------------------------------------------------------------------------------------------------
'''

@multimethod
def get_order(factor: RDOperator):
    """
    Determines the order of a quantum operator.
    
    Parameters
    ----------
    factor : RDOperator
        The quantum operator to evaluate.
    
    Returns
    -------
    tuple
        The order and the classification of the operator ('finite' or 'infinite').
    """
    return 0, ('finite', None)

@multimethod
def get_order(factor: BosonOp):
    """
    Determines the order of a bosonic operator.
    
    Parameters
    ----------
    factor : BosonOp
        The bosonic operator to evaluate.
    
    Returns
    -------
    tuple
        The order and the classification ('infinite') with its associated key.
    """
    key = Dagger(factor)*factor if factor.is_annihilation else factor * Dagger(factor)
    return 0, ('infinite', key)

@multimethod
def get_order(factor: Union[int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """
    Determines the order of basic numeric types.
    
    Parameters
    ----------
    factor : Union[int, float, complex, etc.]
        A numeric factor.
    
    Returns
    -------
    tuple
        The order and its classification ('other').
    """
    return 0, ('other', None)

@multimethod
def get_order(factor: Pow):
    """
    Determines the order of a power expression.
    
    Parameters
    ----------
    factor : Pow
        A power expression to evaluate.
    
    Returns
    -------
    tuple
        The order and its classification.
    """
    base_order, (o_type, o_key) = get_order(factor.base)
    
    return base_order * factor.exp, (o_type, o_key)
    

@multimethod
def get_order(factor: Symbol):
    """
    Determines the order of a symbolic variable.
    
    Parameters
    ----------
    factor : Symbol
        The symbolic variable to evaluate.
    
    Returns
    -------
    tuple
        The order and its classification ('other').
    """
    if isinstance(factor, RDSymbol):
        return factor.order, ('other', None)
    return 0, ('other', None)

def group_by_order(expr):
    """
    Groups terms in an expression by their order, separating finite and infinite terms.
    
    Parameters
    ----------
    expr : Expr
        The expression to group.
    
    Returns
    -------
    dict
        A dictionary mapping orders to terms in the expression.

    """
    terms = expr.expand().as_ordered_terms()        # Expand and get ordered terms from the expression to group
    result = {} 
    for term in terms:                            # Iterate over the terms to group them by order
        order = 0                              # Initialize the order for the current term
        factors = term.as_ordered_factors()   # Get the factors of the term
        result_dict = {'other': [], 'finite': [], 'infinite': {}}   # Initialize the result dictionary. Finite contains the finite operators, infinite contains the infinite operators and other contains the rest of the factors.
        for factor in factors:
            factor_order, (factor_type, factor_key) = get_order(factor)  # Get the order and classification of the factor
            order += factor_order   # Add the factor order to the current term order
            if factor_type == 'infinite':   # If the factor is infinite, add it to the infinite dictionary
                result_dict[factor_type][factor_key] = result_dict[factor_type].get(factor_key, []) + [factor]  # Add the factor to the corresponding key
                continue
            result_dict[factor_type].append(factor) # Add the factor to the corresponding type

        result[order] = result.get(order, []) + [result_dict]   # Add the term to the corresponding order in the result dictionary
    return result

'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                        COUNT BOSONS
    TO-DOS:
        [ ] This can be parallelized over the terms.
---------------------------------------------------------------------------------------------------------------------------------------                                                        
'''

@multimethod
def get_count_boson(factor: BosonOp):
    """
    Returns the count of bosons for a given bosonic operator.
    
    Parameters
    ----------
    factor : BosonOp
        The bosonic operator.
    
    Returns
    -------
    int
        The boson count (+1 or -1 depending on creation/annihilation).
    """
    return (-1)**factor.is_annihilation

@multimethod
def get_count_boson(factor: Pow):
    """
    Returns the boson count for a power of a bosonic operator.
    
    Parameters
    ----------
    factor : Pow
        A power expression of a bosonic operator.
    
    Returns
    -------
    int
        The boson count.
    """
    return get_count_boson(factor.base) * factor.exp

@multimethod
def get_count_boson(factor: One):
    """
    Returns zero boson count for a trivial factor.
    
    Parameters
    ----------
    factor : One
        The trivial factor.
    
    Returns
    -------
    int
        The boson count (always 0).
    """
    return 0

def count_bosons(factors:dict, structure={}):
    """
    Counts the number of bosons in a given set of factors.
    
    Parameters
    ----------
    factors : dict
        A dictionary of factors grouped by subspace.
    
    structure : dict, optional
        A dictionary mapping subspaces to indices (default is an empty dictionary).
    
    Returns
    -------
    ndarray
        An array containing the boson count for each subspace.
    """
    result = np_zeros(len(structure))
    for subspace, operators in factors.items():
        # It can be optimized 
        # factors = {subspace: [factor1, factor2, ...]}
        for operator in operators:
            result[structure[subspace]] += get_count_boson(operator)
    return result

def count_bosonic_subspaces(H):
    """
    Counts the number of distinct bosonic subspaces in a Hamiltonian.
    
    Parameters
    ----------
    H : Expr
        The Hamiltonian expression.
    
    Returns
    -------
    dict
        A dictionary mapping each subspace to a unique index.
    """
    count = 0
    structure = {}
    for a in H.atoms(BosonOp):
        key = Dagger(a)*a if a.is_annihilation else a * Dagger(a)   # Get the key for the bosonic operator
        if key in structure:    # If the key is already in the structure, continue
            continue
        structure[key] = count  # Add the key to the structure with the current count
        count += 1  # Increment the count
    
    # The structure is a dictionary with the keys being the bosonic operators and the values being the corresponding indices for a unique subspace.
    return structure


'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                        CREATE MULGROUPS
    TO-DOS:
        [ ] This can be parallelized over the terms.
---------------------------------------------------------------------------------------------------------------------------------------                                                        
'''

def domain_expansion(term:dict, structure={}, subspaces=None):
    """
    Returns a corresponding MulGroup for a given term.

    Parameters
    ----------
    term : dict   {'other':[], 'finite':[], 'infinite':[]}
        A dictionary with keys 'other', 'finite', and 'infinite' representing the respective parts of the term.
    structure : dict, optional
        A dictionary mapping subspaces to indices (default is an empty dictionary).
    subspaces : list, optional
        A list of subspaces to consider (default is None).

    Returns
    -------
    tuple
        A tuple containing the resulting MulGroup and a boolean indicating if is diagonal.
    """
    
    delta = count_bosons(term['infinite'], structure) if structure != {} else np_array([0])                                     # Count the bosons in the infinite part if there is no structure, return 0 (Diagonal operator on the infinite part)
    is_infinite_diagonal = np_all(delta == 0)                                                                            # Check if the infinite part is diagonal
    infinite_operators_array = np_ones(len(structure), dtype=object) if structure != {} else np_array([1])                # Create an array of ones with the length of the structure if there is a structure, otherwise return 1 (Identity operator on the infinite part)
    
    for number_op, operators in term['infinite'].items():                                                              # Iterate over the infinite operators
        infinite_operators_array[structure[number_op]] = Mul(*operators)                                          # Add the operator to the corresponding index in the infinite operators array

    Ns = np_array(list(structure.keys()))                                                                            # Get the keys of the structure as an array

    if not subspaces:
        return MulGroup(Mul(*term['other']), infinite_operators_array, delta, Ns), is_infinite_diagonal                  # Return the MulGroup with the other factors, infinite operators, boson count, and subspaces

    finite_operators = {subspace.name : eye(subspace.dim) for subspace in subspaces}                              # Create a dictionary with the subspaces as keys and the identity matrix as values
    other_factors = Mul(*term['other'])                                                                             # Get the other factors of the term

    for operator in term['finite']:
        subspace = operator.subspace                                                                               # Get the subspace of the operator
        finite_operators[subspace] = operator.matrix                                                            # Add the operator matrix to the corresponding subspace in the finite operators dictionary

    finite_matrix = kronecker_product(*list(finite_operators.values()))                                          # Create the finite matrix by taking the kronecker product of the finite operators
    is_finite_diagonal = finite_matrix.is_diagonal()                                                          # Check if the finite matrix is diagonal


    return MulGroup(other_factors * finite_matrix, infinite_operators_array, delta, Ns), is_infinite_diagonal and is_finite_diagonal

    
'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                APPLY COMMUTATION RELATIONS
    TO-DOS:
        [x] This can be optized with general rules for commutation relations.
        [ ] This can be parallelized over the terms.
        [ ] Can we optimize the while loop, maybe using wildcards?
---------------------------------------------------------------------------------------------------------------------------------------                                                        
'''

def apply_commutation_relations(expr:Expression, commutation_relations:dict):
    """
    Applies commutation relations to the infinite part of a given expression.

    Parameters
    ----------
    expr : Expression
        The expression to which commutation relations are applied.
    commutation_relations : dict
        A dictionary of commutation relations to apply.

    Returns
    -------
    Expression
        A new expression after applying the commutation relations.
    """
    mul_groups = expr.expr
    result = Expression()
    for group in mul_groups:                        # Iterate over the MulGroups in the expression
        inf = group.inf                            # Get the infinite part of the MulGroup
        fn = group.fn                           # Get the function of the MulGroup
        delta = group.delta                    # Get the boson count of the MulGroup
        inf_new = np_vectorize(lambda x: x.subs(commutation_relations).expand() if isinstance(x, Expr) else x, otypes=[object])(inf)    # Apply the commutation relations to the infinite part

        while np_any(inf_new != inf):
            # Can we optimize this?
            inf = inf_new
            inf_new = np_vectorize(lambda x: x.subs(commutation_relations).expand() if isinstance(x, Expr) else x, otypes=[object])(inf)    # Apply the commutation relations to the infinite part

        inf_terms = np_vectorize(lambda x: x.as_ordered_terms() if isinstance(x, Expr) else [x], otypes=[np_ndarray])(inf)  # Get the terms of the infinite part
        product_terms = product(*inf_terms) # Get the product of the terms
        
        for new_inf in product_terms:
            coeff_inf_array = np_vectorize(lambda x: list(x.as_coeff_Mul()) if isinstance(x, Expr) else [x, 1], otypes=[object])(new_inf)   # Get the coefficient and the term of the infinite part
            coeff = Mul(*[coeff for coeff, _ in coeff_inf_array])   # Get the coefficient of the infinite part
            inf = np_array([term for _, term in coeff_inf_array])   # Get the term of the infinite part

            result += MulGroup(fn * coeff, inf, delta, group.Ns)    # Add the new MulGroup to the result

    return result


'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                FULL DIAGONALIZATION
    TO-DOS:
        [ ] This can be parallelized over the terms.
---------------------------------------------------------------------------------------------------------------------------------------
'''
def separate_diagonal_off_diagonal(expr:Expression):
    """
    Separates diagonal and off-diagonal terms in the given expression.

    Parameters
    ----------
    expr : Expression
        The expression to analyze.

    Returns
    -------
    tuple
        A tuple containing two Expression objects: one for off-diagonal and another for diagonal terms.
    """
    mul_groups = expr.expr
    if len(mul_groups) == 0:
        return Expression(), Expression()
    
    deltas = np_all(np_array([group.delta for group in mul_groups]) == 0, axis=1)

    diagonal_expr = Expression()
    not_diagonal_expr = Expression(mul_groups[np_logical_not(deltas)])

    for group in mul_groups[deltas]:
        diagonal_fn = sp_diag(*group.fn.diagonal())         
        off_diagonal_fn = group.fn - diagonal_fn

        diagonal_expr += MulGroup(diagonal_fn, group.inf, group.delta, group.Ns)
        not_diagonal_expr += MulGroup(off_diagonal_fn, group.inf, group.delta, group.Ns)

    return not_diagonal_expr, diagonal_expr


'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                        EXTRA UTILS
    TO-DOS:
        [ ] Check if there is something to do here.
---------------------------------------------------------------------------------------------------------------------------------------
'''

class memoized(object):
    """
    A decorator class for memoization of function results to improve performance by caching outputs.

    Attributes
    ----------
    func : callable
        The function being decorated.
    cache : dict
        A cache dictionary to store computed results.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        """
        Calls the decorated function, returning cached results when available.

        Parameters
        ----------
        args : tuple
            Arguments to pass to the function.

        Returns
        -------
        The return value of the function.
        """
        value = self.cache.get(args)
        if value is not None:
            return value
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

def partitions(n):
    """
    Generates all partitions of the integer n.

    Parameters
    ----------
    n : int
        The integer to partition.

    Returns
    -------
    list
        A list of tuples representing all unique partitions of n.
    """

    # Base case: if n is 0, return a single partition containing only (0,)
    if n == 0:
        return [(0,)]
    
    parts = [(n,)]                      # Start with the partition of n as a single tuple
    for i in range(1, n + 1):           # Start from 1 to avoid infinite recursion with zero
        for p in partitions(n - i):
            parts.append(p + (i,))
    
    return parts

def commutator(A:Union[MulGroup, Expression], B:Union[MulGroup, Expression]) -> Expression:
    """
    Computes the commutator of two operators A and B.

    Parameters
    ----------
    A : Expression
        The first operator.
    B : Expression
        The second operator.

    Returns
    -------
    Expression
        The resulting commutator.
    """
    return (A * B - B * A).simplify()


def display_dict(dictionary):
    """
    Displays a dictionary of LaTeX expressions in an IPython environment.

    Parameters
    ----------
    dictionary : dict
        A dictionary with keys and values to be displayed.
    """
    for key, value in dictionary.items():
        display(Math(f"{latex(key)} : {latex(value)}"))

def group_by_operators(expr):
    """
    Groups terms in the given expression by quantum operators.

    Parameters
    ----------
    expr : Expression
        The expression to analyze.

    Returns
    -------
    dict
        A dictionary mapping operator terms to their coefficients.
    """
    expr = expr.expand()
    result_dict = {}

    operators = list(expr.atoms(BosonOp)) + list(expr.atoms(RDOperator))
    terms = expr.as_ordered_terms()
    factors_of_terms = [term.as_ordered_factors() for term in terms]

    has_op = lambda o: any([o.has(op) for op in operators])

    for term in factors_of_terms:
        result_term = 1
        result_coeff = 1
        for factor in term:
            if isinstance(factor, Pow):
                base, exp = factor.as_base_exp()
                if has_op(base) and exp > 0:
                    result_term *= base**exp
                    continue
                result_coeff *= factor
                continue
            if has_op(factor):
                result_term *= factor
                continue
            result_coeff *= factor

        result_dict[result_term] = result_dict.get(result_term, 0) + result_coeff
    
    return result_dict

def get_perturbative_expression(expr, structure, subspaces=None):
    """
    Constructs a perturbative multipliccative group expansion of the given expression.

    Parameters
    ----------
    expr : Expression
        The expression to expand perturbatively.
    structure : dict
        A dictionary mapping subspaces to indices.
    subspaces : list
        A list of subspaces to consider during expansion.

    Returns
    -------
    dict
        A dictionary mapping orders to their corresponding simplified expressions.
    """
    expr_ordered_dict = group_by_order(expr)

    result : dict[Expression] = {}

    for order in expr_ordered_dict:
        for term in expr_ordered_dict[order]:
            mul_group_term, is_diagonal = domain_expansion(term, structure, subspaces)
            result[order] = (result.get(order, Expression()) + mul_group_term).simplify()

    return result



'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                TRUNCATE INFINITE PART
    
    - This function is deprecated and should be removed if not used.
---------------------------------------------------------------------------------------------------------------------------------------
'''

def get_boson_matrix(is_annihilation, dim):
    """
    Generates the matrix representation for a bosonic operator.

    Parameters
    ----------
    is_annihilation : bool
        Indicates whether the operator is an annihilation operator.
    dim : int
        The dimension of the matrix to generate.

    Returns
    -------
    Matrix
        A matrix representation for the bosonic operator.
    """
    matrix = sp_zeros(dim, dim)
    if is_annihilation:
        for i in range(1, dim):
            matrix[i-1, i] = sp_sqrt(i)
        return matrix

    for i in range(1, dim):
        matrix[i, i-1] = sp_sqrt(i)
    
    return matrix


@ multimethod
def get_matrix(H: RDOperator, list_subspaces):
    return kronecker_product(*[H.matrix if H.subspace == subspace else eye(dim) for subspace, dim in list_subspaces])

@ multimethod
def get_matrix(H: BosonOp, list_subspaces):
    # list_subspaces : [[subspace, dim], ...]

    return kronecker_product(*[get_boson_matrix(H.is_annihilation, dim) if H.name == subspace else eye(dim) for subspace, dim in list_subspaces])

@ multimethod
def get_matrix(H: Union[Symbol, RDSymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational], list_subspaces):

    return H * kronecker_product(*[eye(dim) for subspace, dim in list_subspaces])

@multimethod
def get_matrix(H: Pow, list_subspaces):
    base, exp = H.as_base_exp()
    return get_matrix(base, list_subspaces) ** exp

@ multimethod
def get_matrix(H: Expr, list_subspaces):
    # list_subspaces : [[subspace, dim], ...]
    H = H.expand()
    result_matrix = sp_zeros(Mul(*[dim for subspace, dim in list_subspaces]))

    terms = H.as_ordered_terms()

    for term in terms:
        term_matrix = 1
        factors = term.as_ordered_factors()
        for factor in factors:
            factor_matrix = get_matrix(factor, list_subspaces)
            term_matrix = term_matrix * factor_matrix
        result_matrix += term_matrix
    return result_matrix
