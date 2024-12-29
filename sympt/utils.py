
"""
Title: Utils for sympt package
Date: 24 December 2024
Authors:
- Giovanni Francesco Diotallevi
- Irving Leander Reascos Valencia

DOI: https://doi.org/10.48550/arXiv.2412.10240

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
17. 'T': Generates list of length-n partitions whose sum is fixed by input parameter.
18. 'P': Generates union of Ts whose sum is less or equal than provided parameter.

Dependencies:
-------------
ipython==8.28.0
multimethod==1.12
numpy==2.1.2
sympy==1.13.3
"""


# Standard library imports
from itertools import product, permutations
from typing import Union

# Third-party imports
from IPython.display import display, Math
from multimethod import multimethod
from numpy import (any as np_any, all as np_all, array as np_array,
                   ndarray as np_ndarray, logical_not as np_logical_not,
                   ones as np_ones, vectorize as np_vectorize,
                   zeros as np_zeros, nonzero as np_nonzero,
                   prod as np_prod, block as np_block)
from sympy import (Expr, Mul, Add, Pow, Symbol, Matrix, exp, latex, diag as sp_diag,
                   cos, sin, factor_terms as sp_factor_terms, conjugate, 
                   factorial as sp_factorial, Rational as sp_Rational, binomial as sp_binomial,
                   Mul as sp_Mul)
from sympy.core.numbers import (
    Float, Half, ImaginaryUnit, Integer, One, Rational, Pi)
from sympy.physics.quantum import Dagger, Operator
from sympy.physics.quantum.boson import BosonOp
from sympy.simplify import nsimplify as sp_nsimplify, simplify as sp_simplify

# Local application/library imports
from .classes import MulGroup, RDSymbol, RDOperator, Expression, get_matrix, t

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
    key = Dagger(factor) * \
        factor if factor.is_annihilation else factor * Dagger(factor)
    return 0, ('infinite', key)

@multimethod
def get_order(factor: Union[exp, cos, sin]):
    """
    Determines the order of a sine expression.

    Parameters
    ----------
    factor : sin
        The sine expression to evaluate.

    Returns
    -------
    tuple
        The order and its classification ('other').
    """
    order, (o_type, o_key) = get_order(factor.args[0]) # Taylor expansion of sin(x) = x - x^3/3! + x^5/5! - ...
    if o_type != 'other':
        raise ValueError(f"Functions of operators are not yet supported.")
    if isinstance(factor, sin):
        return order, ('other', None)
    return 0, ('other', None)

@multimethod
def get_order(factor: Union[int, float, complex, Integer, Float, ImaginaryUnit, One, Rational, Half, Pi]):
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
def get_order(factor: conjugate):
    """
    Determines the order of a conjugate expression.

    Parameters
    ----------
    factor : conjugate
        The conjugate expression to evaluate.

    Returns
    -------
    tuple
        The order and its classification ('other').
    """
    factor = factor.args[0]
    if factor.has(Operator):
        raise ValueError(f"Conjugate of operators are not yet supported.")
    return get_order(factor)

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


@multimethod
def get_order(expr: Expr):
    """
    Determines the order of an expression.

    Parameters
    ----------
    expr : Expr
        The expression to evaluate.

    Returns
    -------
    tuple
        The order and its classification ('other').
    """
    ops = list(expr.atoms(BosonOp) | expr.atoms(RDOperator))
    if len(ops) > 0:
        raise ValueError(
            f"The Hamiltonian contains non-integer or non-positive powers of the operators {ops}.")

    if isinstance(expr, Mul):
        return sum([get_order(op)[0] for op in expr.args]), ('other', None)

    if isinstance(expr, Add):
        orders = set([get_order(op)[0] for op in expr.args])
        if len(orders) > 1:
            raise ValueError(
                f"The Hamiltonian contains non-integer or non-positive powers of the sum of terms with different orders: {orders}. Thus, the order of {expr} is ambiguous.")

        return orders.pop(), ('other', None)


@multimethod
def group_by_order(expr:Matrix):
    """
    Groups terms in a matrix by their order, separating finite and infinite terms.

    Parameters
    ----------
    expr : Matrix
        The matrix to group.

    Returns
    -------
    dict
        A dictionary mapping orders to terms in the matrix.

        {order: [{'other': [other_factors], 'finite': [finite_operators], 'infinite': {infinite_operators}}, ...]}
    """
    result = {}
    expr_non_zeros = np_nonzero(expr)
    for i, j in zip(*expr_non_zeros):
        projector = Matrix.zeros(expr.rows, expr.cols)
        projector[i, j] = 1
        terms = group_by_order(expr[i, j])
        for order, term in terms.items():
            for factor in term:
                factor['other'].append(projector)
            result[order] = result.get(order, []) + term
    return result

@multimethod
def group_by_order(expr:Expr):
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

        {order: [{'other': [other_factors], 'finite': [finite_operators], 'infinite': {infinite_operators}}, ...]}

    """
    terms = expr.expand().as_ordered_terms(
    )        # Expand and get ordered terms from the expression to group
    result = {}
    for term in terms:                            # Iterate over the terms to group them by order
        order = 0                              # Initialize the order for the current term
        factors = term.as_ordered_factors()   # Get the factors of the term
        # Initialize the result dictionary. Finite contains the finite operators, infinite contains the infinite operators and other contains the rest of the factors.
        result_dict = {'other': [], 'finite': [], 'infinite': {}}
        for factor in factors:
            factor_order, (factor_type, factor_key) = get_order(
                factor)  # Get the order and classification of the factor
            order += factor_order   # Add the factor order to the current term order
            if factor_type == 'infinite':   # If the factor is infinite, add it to the infinite dictionary
                result_dict[factor_type][factor_key] = result_dict[factor_type].get(
                    factor_key, []) + [factor]  # Add the factor to the corresponding key
                continue
            # Add the factor to the corresponding type
            result_dict[factor_type].append(factor)

        # Add the term to the corresponding order in the result dictionary
        result[order] = result.get(order, []) + [result_dict]
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
def get_count_boson(factor: conjugate):
    """
    Returns the boson count for a conjugate expression.

    Parameters
    ----------
    factor : conjugate
        The conjugate expression.

    Returns
    -------
    int
        The boson count.
    """
    factor = factor.args[0]
    if factor.has(Operator):
        raise ValueError(f"Conjugate of operators are not yet supported.")
    return get_count_boson(factor)

@multimethod
def get_count_boson(factor: Union[RDSymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational, Pi, exp]):
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


def count_bosons(factors: dict, structure={}):
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
        key = Dagger(a)*a if a.is_annihilation else a * \
            Dagger(a)   # Get the key for the bosonic operator
        if key in structure:    # If the key is already in the structure, continue
            continue
        # Add the key to the structure with the current count
        structure[key] = count
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


def domain_expansion(term: dict, structure={}, subspaces=None):
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

    # Count the bosons in the infinite part if there is no structure, return 0 (Diagonal operator on the infinite part)
    delta = count_bosons(term['infinite'], structure) if structure != {
    } else np_array([0])
    # Check if the infinite part is diagonal
    is_infinite_diagonal = np_all(delta == 0)
    # Create an array of ones with the length of the structure if there is a structure, otherwise return 1 (Identity operator on the infinite part)
    infinite_operators_array = np_ones(
        len(structure), dtype=object) if structure != {} else np_array([1])

    # Iterate over the infinite operators
    for number_op, operators in term['infinite'].items():
        # Add the operator to the corresponding index in the infinite operators array
        infinite_operators_array[structure[number_op]] = Mul(*operators)

    # Get the keys of the structure as an array
    Ns = np_array(list(structure.keys()))

    if not subspaces:
        # Return the MulGroup with the other factors, infinite operators, boson count, and subspaces
        return MulGroup(Mul(*term['other']), infinite_operators_array, delta, Ns), is_infinite_diagonal

    finite_matrix = get_matrix(term['finite'], subspaces)  # Get the finite matrix
    # Check if the finite matrix is diagonal
    is_finite_diagonal = finite_matrix.is_diagonal()

    # Get the other factors of the term
    other_factors = Mul(*term['other'])

    return MulGroup(other_factors * finite_matrix, infinite_operators_array, delta, Ns), is_infinite_diagonal and is_finite_diagonal


'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                APPLY COMMUTATION RELATIONS
    TO-DOS:
        [x] This can be optized with general rules for commutation relations.
        [ ] This can be parallelized over the terms.
        [ ] Can we optimize the while loop, maybe using wildcards?
        [ ] Add checkings for the commutation relations, that's why commutation_relations is in a different function.
---------------------------------------------------------------------------------------------------------------------------------------                                                        
'''


def apply_substituitions(expr: Expression, subs: dict):
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
        inf_new = np_vectorize(lambda x: x.subs(subs).expand() if isinstance(x, Expr) else x, otypes=[
                               # Apply the commutation relations to the infinite part
                               object])(inf)

        while np_any(inf_new != inf):
            # Can we optimize this?
            inf = inf_new
            inf_new = np_vectorize(lambda x: x.subs(subs).expand() if isinstance(x, Expr) else x, otypes=[
                                   # Apply the commutation relations to the infinite part
                                   object])(inf)

        inf_terms = np_vectorize(lambda x: x.as_ordered_terms() if isinstance(
            # Get the terms of the infinite part
            x, Expr) else [x], otypes=[np_ndarray])(inf)
        product_terms = product(*inf_terms)  # Get the product of the terms

        for new_inf in product_terms:
            coeff_inf_array = np_vectorize(lambda x: list(x.as_coeff_Mul()) if isinstance(x, Expr) else [
                                           # Get the coefficient and the term of the infinite part
                                           x, 1], otypes=[object])(new_inf)
            # Get the coefficient of the infinite part
            coeff = Mul(*[coeff for coeff, _ in coeff_inf_array])
            # Get the term of the infinite part
            inf = np_array([term for _, term in coeff_inf_array])

            # Add the new MulGroup to the result
            result += MulGroup(fn * coeff, inf, delta, group.Ns)

    return result


def apply_commutation_relations(expr: Expression, commutation_relations: dict):
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
    return apply_substituitions(expr, commutation_relations)


def extract_ns(expr: Expression, structure: dict):
    """
    Extracts the subspaces from the infinite part of a given expression.

    Parameters
    ----------
    expr : Expression
        The expression to extract subspaces from.

    Returns
    -------
    Expression
        A new expression with the subspaces extracted.
    """

    if structure == {}:
        return expr, {}

    Ns = np_array(list(structure.keys()))
    Ads, As = np_array([N.as_ordered_factors() for N in Ns]).T

    ns_comm = dict(zip(Ads**2 * As, (Ns - 1) * Ads))

    return apply_substituitions(expr, ns_comm), ns_comm

'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                FULL DIAGONALIZATION
    TO-DOS:
        [ ] This can be parallelized over the terms.
---------------------------------------------------------------------------------------------------------------------------------------
'''


def separate_diagonal_off_diagonal(expr: Expression):
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

    deltas = np_all(
        np_array([group.delta for group in mul_groups]) == 0, axis=1)
    
    diagonal_expr = Expression()
    not_diagonal_expr = Expression(mul_groups[np_logical_not(deltas)])

    for group in mul_groups[deltas]:
        diagonal_fn = sp_diag(*group.fn.diagonal())
        off_diagonal_fn = (group.fn - diagonal_fn).expand()
        
        diagonal_expr += MulGroup(diagonal_fn, group.inf,
                                  group.delta, group.Ns)
        not_diagonal_expr += MulGroup(off_diagonal_fn,
                                      group.inf, group.delta, group.Ns)

    return not_diagonal_expr, diagonal_expr


'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                        EXTRA UTILS
    TO-DOS:
        [ ] Check if there is something to do here.
---------------------------------------------------------------------------------------------------------------------------------------
'''

def get_block_mask(block_sizes):
    """
    Generate a block structure with specified block sizes.
    
    Parameters:
    -----------
    block_sizes : list of int
        List of block sizes for each section of the matrix
        
    Returns:
    --------
    structure : numpy.ndarray
        The generated block structure matrix
    """
    # functions for generating blocks
    block = lambda i, j: np_ones((i, j))
    zero = lambda i, j: np_zeros((i, j))
    
    
    n = len(block_sizes) # initial block structure
    structure = []
    
    for i in range(n):
        row = []
        for j in range(n):
            # Fill with zeros if i > j (lower triangle)
            if i > j:
                row.append(zero(block_sizes[i], block_sizes[j]))
            # Create block of ones for diagonal
            elif i == j:
                row.append(zero(block_sizes[i], block_sizes[j]))
            # Fill with ones for upper triangle
            else:
                row.append(block(block_sizes[i], block_sizes[j]))
        structure.append(row)
    

    structure = np_block(structure) # Convert to numpy block matrix
    
    structure = structure + structure.T # Make symmetric 
    
    return structure
  
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


def commutator(A: Union[MulGroup, Expression], B: Union[MulGroup, Expression]) -> Expression:
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

def create_nest_commute(Os_dicts, Ss):
    def nest_commute(parts, Os_index):
        if len(parts) == 2:
            Os = Os_dicts[Os_index]
            return commutator(Os.get(parts[0], 0), Ss[parts[1]])
        return commutator(nest_commute(parts[:-1], Os_index), Ss[parts[-1]])
    return memoized(nest_commute)


def display_dict(dictionary):
    """
    Displays a dictionary of LaTeX expressions in an IPython environment.

    Parameters
    ----------
    dictionary : dict
        A dictionary with keys and values to be displayed.
    """
    for key, value in dictionary.items():
        key = key.simplify() if isinstance(key, Expression) else key
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

    operators = list(expr.atoms(BosonOp)) + list(expr.atoms(RDOperator))  + list(expr.atoms(Operator))
    terms = expr.as_ordered_terms()
    factors_of_terms = [term.as_ordered_factors() for term in terms]

    def has_op(o): return any([o.has(op) for op in operators])

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
        result_term = result_term.simplify() if isinstance(
            result_term, Mul) else result_term
        result_dict[result_term] = result_dict.get(
            result_term, 0) + result_coeff

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
    if isinstance(expr, Matrix):
        subspaces = None
    expr_ordered_dict = group_by_order(expr)
    orders = np_array(list(expr_ordered_dict.keys()))

    min_order = min(orders)
    if min_order < 0:
        term_negative = [Mul(*terms['other'])
                         for terms in expr_ordered_dict[min_order]]
        error_message = f"The expression contains the terms ["
        error_message += ", ".join([f"{term}" for term in term_negative])
        error_message += f"] which have a total negative order {min_order}. Something on your definition is maybe wrong. Otherwise, consider redefining the unperturbed Hamiltonian."
        raise ValueError(error_message)

    is_integer_order = orders.astype(int) - orders

    if not np_all(is_integer_order == 0):
        error_message = f"The Hamiltonian contains terms whose total perturbative order is not an integer: "
        non_integer_orders = orders[is_integer_order != 0]
        for order in non_integer_orders:
            term_non_integer = [Mul(*terms['other'])
                                for terms in expr_ordered_dict[order]]
            error_message += f"\nThe terms {term_non_integer} have a total order {order}."
        raise ValueError(error_message)

    result: dict[Expression] = {}
    for order in expr_ordered_dict:
        for term in expr_ordered_dict[order]:
            mul_group_term, is_diagonal = domain_expansion(
                term, structure, subspaces)
            order = int(order)
            result[order] = (result.get(order, Expression()) +
                             mul_group_term).simplify()

    return result

def extract_frequencies(term):
    if not term.has(t):
        return 0
    term = sp_factor_terms(term)
    exponentials_atoms = term.atoms(exp)
    if len(exponentials_atoms) == 0:
        return 0
    exponentials_atoms = [exp_atom for exp_atom in exponentials_atoms if exp_atom.has(t)]
    if len(exponentials_atoms) > 1:
        raise ValueError("The term contains more than one exponential. If you see this error, please report it to the developers.")
    
    exp_arg = exponentials_atoms.pop().args[0]
    return exp_arg.coeff(t)

'''
---------------------------------------------------------------------------------------------------------------------------------------
                                                        Generate Partitions
    TO-DOS:
        [ ] 
---------------------------------------------------------------------------------------------------------------------------------------
'''

def partitions(n):
    """
    Generates all partitions of the integer n. (Mostly used in keeping track of nested commutators)

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

    # Start with the partition of n as a single tuple
    parts = [(n,)]
    for i in range(1, n + 1):           # Start from 1 to avoid infinite recursion with zero
        for p in partitions(n - i):
            parts.append(p + (i,))

    return parts

# Partitions for Least Action (LA) method

def T(order, lenght):
    orders = range(1, order + 1)
    tupples = product(orders, repeat=lenght)
    return [t for t in tupples if sum(t) == order]

def P(order):
    partitions = []
    for i in range(1, order + 1):
        partitions.extend(T(order, i))
    return partitions
