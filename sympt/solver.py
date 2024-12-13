
"""
Title: Solver for sympt package
Date: 13 December 2024
Authors:
- Giovanni Francesco Diotallevi
- Irving Leander Reascos Valencia

DOI: doi.doi.doi

Description:
------------
This module provides classes and functions for solving perturbative expansions and transformations for quantum
systems using the Schrieffer-Wolff transformation technique. It contains utilities to compute commutators,
apply transformations, and obtain the effective Hamiltonian.

Classes:
--------
1. EffectiveFrame: Encapsulates methods for solving the Schrieffer-Wolff transformation for a Hamiltonian.

Functions:
----------
1. Denominator: Computes the energy denominator for off-diagonal elements of the Hamiltonian.
2. get_S: Solves for the anti-Hermitian matrix S in the Schrieffer-Wolff transformation.

Dependencies:
-------------
ipython==8.28.0
multimethod==1.12
numpy==2.1.2
sympy==1.13.3
tabulate==0.9.0
tqdm==4.66.5

- utils.py
- classes.py

To Do:
------
[ ] Denominator function can be memoized for better performance.
[ ] Implement parallelization for the computation of the energy denominator.
[ ] Denominator can be changed to handle time-dependent perturbations.

[ ] S function can be optimized by using the Hermitian property of the EquationtoSolve and the anti-Hermitian property of S.
[ ] Implement parallelization for the computation of the anti-Hermitian Operator S.

[ ] Prepare the result method can be optimized for better performance by parallelizing the conversion to matrix form.
"""


# Local application/library imports
from .classes import *
from .utils import *

# Third-party imports
from tqdm import tqdm, trange
from tabulate import tabulate
from sympy import (Rational as sp_Rational, factorial as sp_factorial,
                   nsimplify as sp_nsimplify, simplify as sp_simplify,
                   Add as sp_Add, Matrix as sp_Matrix)
from numpy import any as np_any
# import deep copy
from copy import copy

from .logging_config import setup_logger

logger = setup_logger(__name__)

def Denominator(H0_expr, uu, vv, Delta):
    """
    Computes the energy denominator for off-diagonal elements of the Hamiltonian during the Schrieffer-Wolff transformation.

    Parameters
    ----------
    H0_expr : Expression
        The zeroth-order Hamiltonian.
    uu : int
        The row index of the matrix element.
    vv : int
        The column index of the matrix element.
    Delta : ndarray
        The shift between different bosonic states.

    Returns
    -------
    Expr
        The energy difference between the matrix elements uu and vv after applying Delta.
    """
    E = 0
    for mul_group in H0_expr.expr:
        Ns = np_vectorize(sp_simplify)(mul_group.inf)
        js = np_zeros(Ns.shape)

        for eta, n in enumerate(Ns):
            if n != 1:
                js[eta] = 1
            if not isinstance(n, Pow):
                continue
            Ns[eta] = n.base
            js[eta] = n.exp

        Epsilon_j = mul_group.fn
        E_mu = Epsilon_j[uu, uu]
        E_nu = Epsilon_j[vv, vv]
        E += sp_nsimplify((E_mu * Mul(*(Ns**js)).simplify() -
                          E_nu * Mul(*((Ns - Delta)**js)).simplify()))
        
    return E.expand().simplify()


def get_S(H0_expr, equation_to_solve, correct_denominator=False):
    """
    Solves for the anti-Hermitian operator S in the Schrieffer-Wolff transformation.

    Parameters
    ----------
    H0_expr : Expression
        The zeroth-order Hamiltonian.
    equation_to_solve : Expression
        The expression for the off-diagonal terms in the perturbative expansion.

    Returns
    -------
    Expression
        The anti-Hermitian Operator S used in the Schrieffer-Wolff transformation.
    """
    if not np_any(equation_to_solve.expr):
        return Expression()
    S = Expression()
    for mul_group in equation_to_solve.expr:
        Delta = mul_group.delta
        S_mat = MutableDenseMatrix(mul_group.fn)
        S_mat_non_zeros = np_nonzero(S_mat)

        for uu, vv in zip(*S_mat_non_zeros):
            # This can be optimized by using the fact that the EquationtoSolve is Hermitian and S is anti-Hermitian
            # This is a good candidate for parallelization
            denom = Denominator(H0_expr, uu, vv, Delta)

            if denom == 0 and not correct_denominator:
                if uu == vv and Delta == 0:
                    raise ValueError(f'S contains diagonal elements. If you saw this message, please contact the developers.')
                raise ValueError(f'It is impossible to decouple the state with index {int(uu)} (Delta {Delta}) from state with index {int(vv)} (Delta {Delta}) because they are degenerate (see H0)')
            elif not correct_denominator:
                S_mat[uu, vv] /= denom
            else:
                corrected_S = 0
                for term in S_mat[uu, vv].expand().as_ordered_terms():
                    freq_term = (extract_frequencies(term) * (-I  * hbar)).cancel()  # We assume that the argument for the exponential is in the form of exp(i/hbar Energy t)
                    corrected_S += term / (denom + freq_term)
                S_mat[uu, vv] = corrected_S


        S += MulGroup(S_mat, mul_group.inf, mul_group.delta, mul_group.Ns)
    return S


class EffectiveFrame:
    """
    A class representing the effective frame obtained via the Schrieffer-Wolff transformation.

    Attributes
    ----------
    H_input : Expression or Matrix
        The zeroth-order Hamiltonian.
    V_input : Expression or Matrix
        The perturbative interaction Hamiltonian.
    subspaces : list, optional
        The list of subspaces to consider in the transformation.

    Methods
    -------
    solve(max_order=2, full_diagonalization=False, mask=None):
        Solves for the effective Hamiltonian using the Schrieffer-Wolff transformation.

    get_H(return_operator_form=True):
        Returns the effective Hamiltonian in either operator form or matrix form.

    rotate(expr, max_order=None, return_operator_form=True):
        Rotates a given expression according to the computed transformation S.
    """

    def __init__(self, H, V=None, subspaces=None, symbol_values=None, verbose=True):
        """
        Initializes the EffectiveFrame object.

        Parameters
        ----------
        H : Expression
            The zeroth-order Hamiltonian.
        V : Expression
            The perturbative interaction Hamiltonian (default is None).
        subspaces : list, optional
            A list of subspaces to consider (default is None).
        """
        v = V if V is not None else sp_zeros(*H.shape) if isinstance(H, sp_Matrix) else 0
        sint_cost_dict = {k : - I * Rational(1,2) * (exp(I * k.args[0]) - exp(-I * k.args[0])) for k in (H + v).atoms(sin) if k.has(t)}
        sint_cost_dict.update({k : Rational(1,2) * (exp(I * k.args[0]) + exp(-I * k.args[0])) for k in (H + v).atoms(cos) if k.has(t)})

        self.H_input = H.subs(sint_cost_dict)
        self.V_input = V.subs(sint_cost_dict) if V is not None else V

        self.symbol_values = symbol_values
        self.__do_substitute = symbol_values is not None

        del v
        del sint_cost_dict

        self.subspaces = subspaces
        self.__return_form = 'operator'
        self.__structure = count_bosonic_subspaces(self.H_input)
        self.commutation_relations = self.get_commutation_relations()

        self.verbose = verbose
        if not verbose:
            logger.setLevel('ERROR')

        self.formatter()

        logger.info('The EffectiveFrame object has been initialized successfully.')

    def get_commutation_relations(self):
        if hasattr(self, 'commutation_relations'):
            return self.commutation_relations
        Ns = np_array(list(self.__structure.keys()))
        if Ns.shape[0] == 0:
            return {}
        keys = np_vectorize(lambda x: Mul(
            *(x.as_ordered_factors()[::-1])))(Ns)  # a*ad

        # Compute the commutation relations for the bosonic subspaces (a*ad = ad * a + 1)
        commutation_relations = dict(
            zip(keys, Ns + 1)) if self.__structure != {} else {}
        return commutation_relations

    def formatter(self):
        if self.H_input.has(RDOperator) and self.subspaces is None:
            raise ValueError(
                'Subspaces must be provided when the Hamiltonian contains RDOperator objects.')

        if isinstance(self.H_input, Matrix):
            logger.info('Creating the EffectiveFrame object with matrix form.')
            if self.V_input is not None and not isinstance(self.V_input, Matrix):
                raise ValueError('The type of V and H must be the same.')

            if self.V_input is not None and self.H_input.shape != self.V_input.shape:
                raise ValueError('The shapes of V and H must be the same.')

            if self.subspaces is None:
                self.__return_form = 'matrix'
                return

            total_dim = Mul(*[subspace.dim for subspace in self.subspaces])
            if total_dim != self.H_input.shape[0]:
                raise ValueError(
                    f'The dimension of the finite subspace {self.__composite_basis.dim} must be the same as the Hamiltonian {self.H_input.shape[0]}.')

    def __checks_and_prepare_solver(self, method, mask, max_order=2):

        method = method.upper()
        if method not in ['SW', 'FD', 'BD', 'LA', 'ACE']:
            raise ValueError('Invalid method. Please choose one of the following: SW, FD, BD, LA, ACE.')
        
        if method == 'BD':
            method = 'LA'

        self.__H_old = copy(self.H_input)
        self.__V_old = copy(self.V_input)

        # Check if the perturbative interaction is provided
        if method == 'SW' and self.V_input is None:
            # Raise an error if the perturbative interaction is not provided
            raise ValueError(
                'The perturbative interaction must be provided for the regular Schrieffer-Wolff transformation')

        # Check if the mask is used in the full diagonalization mode
        if method == 'FD' and mask is not None:
            mask = None
            # If the mask is used in the full diagonalization mode, it will be ignored
            logger.info(
                'The mask is not used in the full diagonalization mode and it will be ignored')

        # Check if the perturbative interaction will be added to the full Hamiltonian
        if self.V_input is not None and method != 'SW':
            logger.info('The perturbative interaction will be added to the full Hamiltonian')
            # Add the perturbative interaction to the Hamiltonian
            self.__H_old = (self.H_input + self.V_input).expand()
            # Set the perturbative interaction to zero
            self.__V_old = S.Zero

        if method != 'FD' and method != 'SW' and mask is not None:
            # Add the Hermitian conjugate and the blocks to the mask. This ensures that the mask is Hermitian
            mask = mask + mask.hermitian()
            if isinstance(mask, Block):
                mask = Blocks(np_array([mask]), subspaces=mask.subspaces)
            # Check if the order of the subspaces in the mask is the same as the order of the subspaces in the Hamiltonian
            if mask.subspaces is not None and mask.subspaces != [subspace.name for subspace in self.subspaces]:
                raise ValueError(
                    'The order of the subspaces in the mask must be the same as the order of the subspaces in the Hamiltonian')
            
            # Add the structure of the bosonic subspaces to the mask
            mask.add_structure(self.__structure)

        # Compute the perturbative expression for the Hamiltonian
        Hs_aux = get_perturbative_expression(
            self.__H_old, self.__structure, self.subspaces)
        
        # If the full diagonalization or mask routine is used
        self.__Vs = {}
        self.__Hs = {}

        # If the regular Schrieffer-Wolff transformation is used
        if method == 'SW':
            # Set the Hamiltonians to the perturbative expressions
            self.__Hs = Hs_aux
            # Compute the perturbative expression for the perturbative interaction
            self.__Vs = get_perturbative_expression(
                self.__V_old, self.__structure, self.subspaces)
            tmp_H0_expr = self.__Hs.get(0) #Expression of zeroth order hamiltonian
            if not tmp_H0_expr: # check if Hamiltonian has zeroth order term
                raise ValueError("The provided Hamiltonian contains no diagonal zeroth order term.")
            for mulgroup in tmp_H0_expr.expr:
                if not mulgroup.is_diagonal(): # if mulgroup is not diagonal
                    raise ValueError("The provided Hamiltonian contains zeroth order terms outside the diagonal. Rotate your system to avoid this behaviour (note that by definition H0 must be solvable)")
            if self.__Vs.get(0) is not None:
                # Check if the zeroth order of the perturbative interaction is zero
                raise ValueError(
                    f'The zeroth order of the perturbative interaction is not zero, but it is instead: {self.__Vs.get(0)}')
            for key in list(self.__Vs.keys()): # iterating through orders of Vs
                tmp_Vs = self.__Vs[key]
                for mulgroup in tmp_Vs.expr:
                    if mulgroup.is_diagonal():
                        raise ValueError("V cannot contain diagonal elements.")

        else:
            if method == 'LA':
                self.__Hs_aux = Hs_aux
            # Iterate over the perturbative expressions for the Hamiltonian
            for h_order, h_expr in Hs_aux.items():
                # If the order is zero
                if h_order == 0:
                    # Separate the diagonal and off-diagonal terms of the Hamiltonian
                    new_vk, new_hk = separate_diagonal_off_diagonal(h_expr)
                    # If the diagonal terms are not zero
                    if new_vk.expr.shape[0] != 0:
                        # Check if the zeroth order of the Hamiltonian is diagonal
                        raise ValueError(
                            'The zeroth order of the Hamiltonian should be diagonal')
                    # Set the Hamiltonian to the off-diagonal terms
                    self.__Hs[h_order] = new_hk
                    continue

                # If the mask is used
                if method != 'FD' and method != 'LA' and mask is not None:
                    # Apply the mask to the perturbative expression
                    new_vk, new_hk = mask.apply_mask(h_expr)
                # If the full diagonalization routine is used
                else:
                    # Separate the diagonal and off-diagonal terms of the Hamiltonian
                    new_vk, new_hk = separate_diagonal_off_diagonal(h_expr)

                self.__Hs[h_order] = new_hk
                self.__Vs[h_order] = new_vk

        # Apply the commutation relations to the zeroth-order Hamiltonian
        H0_expr = apply_commutation_relations(
        self.__Hs.get(0) + Expression(), self.commutation_relations).simplify()
        if H0_expr.is_time_dependent:
            raise ValueError('The zeroth order of the Hamiltonian must be time-independent.')
        self.__do_time_dependent = np_any([v.is_time_dependent for k, v in self.__Hs.items() if k != 0]) or np_any([v.is_time_dependent for k, v in self.__Vs.items() if k != 0])

        if self.__do_time_dependent:
            if method == 'LA':
                raise NotImplementedError('Time-dependent perturbations are not supported in the least action method yet.')

            #### checking if time periodicity is periodic ####
            for tmp_H_order in self.__Hs.values():
                if tmp_H_order.is_time_dependent:
                    mulgroups = tmp_H_order.expr
                    for mulgroup in mulgroups:
                        if mulgroup.is_time_dependent:
                            if not mulgroup.is_t_periodic():
                                raise ValueError("Non periodic time dependencies are not yet supported")
            for tmp_V_order in self.__Vs.values():
                if tmp_V_order.is_time_dependent:
                    mulgroups = tmp_V_order.expr
                    for mulgroup in mulgroups:
                        if mulgroup.is_time_dependent:
                            if not mulgroup.is_t_periodic():
                                raise ValueError("Non periodic time dependencies are not yet supported")
            ###################################################
            freqs_orders = [get_order(exponential.args[0])[0] for exponential in (self.__H_old + self.__V_old).atoms(exp) if exponential.has(t)]

            if len(set(freqs_orders)) > 1:
                raise ValueError('The Hamiltonian contains multiple frequencies with different orders. This is not supported yet.')
            
            self.__frequency_order = freqs_orders[0]
            if int(self.__frequency_order) - self.__frequency_order != 0 or self.__frequency_order < 0:
                raise ValueError('The driving frequency order must be a positive integer or zero.')
            
            self.__frequency_order = int(self.__frequency_order)
            self.__is_frequency_perturbative = self.__frequency_order > 0

        if self.__do_substitute:
            logger.info('Substituting the symbol values in the Hamiltonian and perturbative interactions.')
            self.__Hs = {k: v.subs(self.symbol_values) for k, v in self.__Hs.items()}
            self.__Vs = {k: v.subs(self.symbol_values) for k, v in self.__Vs.items()}
            if method == 'LA':
                self.__Hs_aux = {k: v.subs(self.symbol_values) for k, v in self.__Hs_aux.items()}
            
            H0_expr = self.__Hs.get(0)

        # Compute the factorials for the perturbative orders
        factorials = [sp_Rational(1, sp_factorial(k))
                      for k in range(0, max_order + 3)]
        
        # Extract the number operators from  the zeroth-order Hamiltonian
        H0_expr, self.__ns = extract_ns(H0_expr, self.__structure)
        H0_expr = H0_expr.simplify()

        # Initialize the dictionary to store the anti-Hermitian operator S for each order
        self.__S = {}
        self.__dtSs = {}

        self.__Hs_final = {0 : H0_expr}

        partitions_orders = np_vectorize(partitions, otypes=[np_ndarray])(range(1, max_order + 1))

        return H0_expr, mask, factorials, partitions_orders
    
    def __SW_Bk_Hk(self, **kwargs):

        order = kwargs['order']
        key = kwargs['key']
        factorial_n, factorial_n_1 = kwargs['factorials']
        is_nestedness_even = kwargs['is_nestedness_even']
        nest_commute = kwargs['nest_commute']

        # Compute the nested commutator for the regular Schrieffer-Wolff transformation
        self.__B_k = (self.__B_k + nest_commute(key, is_nestedness_even) * factorial_n).simplify()
        # Compute the nested commutator for the regular Schrieffer-Wolff transformation
        self.__Hs_final[order] = (self.__Hs_final.get(order, Expression()) + nest_commute(key, not is_nestedness_even) * factorial_n).simplify()

        if self.__do_time_dependent:
            if is_nestedness_even:
                self.__B_k = (self.__B_k - I * hbar * nest_commute(key, 2) * factorial_n_1).simplify()
            else:
                self.__Hs_final[order] = (self.__Hs_final.get(order, Expression()) - I * hbar * nest_commute(key, 2) * factorial_n_1).simplify()

    def __FD_ACE_Bk_Hk(self, **kwargs):

        order = kwargs['order']
        key = kwargs['key']
        factorial_n, factorial_n_1 = kwargs['factorials']
        is_nestedness_even = kwargs['is_nestedness_even']
        method = kwargs['method']
        mask = kwargs['mask']
        nest_commute = kwargs['nest_commute']

        # Compute the nested commutator for the full diagonalization or mask routine
        new_commutator_odd = nest_commute(
            key, not is_nestedness_even) * factorial_n
        # Compute the nested commutator for the full diagonalization or mask routine
        new_commutator_even = nest_commute(
            key, is_nestedness_even) * factorial_n

        # Compute the nested commutator for the full diagonalization or mask routine
        new_commutator = new_commutator_odd + new_commutator_even
        if self.__do_time_dependent:
            # Compute the nested commutator for the dS/dt term
            new_commutator_dS = nest_commute(key, 2) * factorial_n_1
            new_commutator -= I * hbar * new_commutator_dS

        new_commutator = new_commutator.simplify()

        if method == 'FD':
            # Separate the diagonal and off-diagonal terms of the nested commutator
            new_bk, new_hf = separate_diagonal_off_diagonal(new_commutator)
        else:
            # Apply the mask to the nested commutator
            new_bk, new_hf = mask.apply_mask(new_commutator)
        
        # Add the nested commutator to the operator B_k
        self.__B_k = (self.__B_k + new_bk).simplify()
        # Add the nested commutator to the final Hamiltonian
        self.__Hs_final[order] = (self.__Hs_final.get(order, Expression()) + new_hf).simplify()

    def __LA_solver(self, max_order, mask):

        def subs_zs(theta_vec):
            if len(theta_vec) == 1:
                return self.__Z[theta_vec[0]]
            return subs_zs(theta_vec[:-1]) * self.__Z[theta_vec[-1]]
        
        self.__S = {}
        LA_S = LA_S_generator(function=memoized(subs_zs), mask=mask)
        for order in trange(1, max_order + 1, desc='Computing least-action generators S', disable=not self.verbose):
            Sk = LA_S(order)
            self.__S[order] = apply_commutation_relations(Sk.doit(), self.commutation_relations).simplify()
        
        self.__Hs_final = {k: apply_substituitions(v, self.__ns).simplify() for k, v in self.__rotate(max_order, self.__Hs_aux).items()}

    def solve(self, max_order=2, method='SW', mask=None):
        """
        Solves for the effective Hamiltonian up to the specified order using the Schrieffer-Wolff transformation.

        Parameters
        ----------
        max_order : int, optional
            The maximum perturbative order to solve for (default is 2).
        method : str, optional
            Supported methods:
                - SW  : Regular Schrieffer-Wolff transformation.
                - FD  : Full diagonalization.
                - BD  : Block diagonalization. Default "least action" (LA) method.
                - LA  : Block diagonalization with least action method. (default). Time-dependent perturbations are not supported.
                - ACE : Arbitrary coupling elimination method. For time-dependent perturbations use this method.
        mask : Expression, optional
            A mask expression used for selectively applying transformations (default is None).
        """

        # Prepare the Hamiltonians and perturbative interactions
        H0_expr, mask, factorials, partitions_orders = self.__checks_and_prepare_solver(method, mask, max_order)

        Os_dicts = [self.__Hs, self.__Vs, self.__dtSs]
        nest_commute = create_nest_commute(Os_dicts, self.__S)


        solver_prints = {
            'SW': 'Time Dependent SWT' if self.__do_time_dependent else 'SWT',
            'FD': 'Time Dependent Full Diagonalization' if self.__do_time_dependent else 'Full Diagonalization',
            'ACE': 'Time Dependent Arbitrary coupling elimination' if self.__do_time_dependent else 'Block Diagonalization',
        }

        solver_prints['LA'] = solver_prints['FD']
        solver_prints['BD'] = solver_prints['LA']
        
        # Iterate over the perturbative orders
        for order in trange(1, max_order + 1, desc=f'Performing {solver_prints[method]} for each order', disable=not self.verbose):
            # Compute the partitions for the perturbative order
            set_of_keys = partitions_orders[order - 1]
            # Initialize the operator B_k for the perturbative order
            self.__B_k = Expression()

            # Iterate over the partitions. Eliminate the last partition because it is the term [H0, S] and it is used to compute the operator S
            for key in set_of_keys[:-1]:
                if len(key) == 1:
                    # Does not deppend of full_diagonalization neither on mask
                    Vk = self.__Vs.get(key[0], Expression())
                    # Add the perturbative interaction to the operator B_k (Equation to solve)
                    self.__B_k += Vk
                    # If do_time_dependent
                    if self.__do_time_dependent:
                        # dtSs[order + self.__frequency_order] = Ss.get(order, Expression()).diff(t)
                        self.__B_k -= I * hbar * self.__dtSs.get(order, Expression())

                    # Add the perturbative Hamiltonian to the final Hamiltonian
                    if method != 'LA':
                        self.__Hs_final[key[0]] = self.__Hs_final.get(key[0], Expression()) + self.__Hs.get(key[0], Expression())
                    continue

                # Compute the nestedness of the partition
                nestedness = len(key) - 1
                is_nestedness_even = nestedness % 2 == 0
                # If the regular Schrieffer-Wolff transformation is used

                solver_input = {
                    'order': order,
                    'key': key,
                    'factorials': (factorials[nestedness], factorials[nestedness + 1]),
                    'is_nestedness_even': is_nestedness_even,
                    'nest_commute': nest_commute
                }

                if method == 'SW':
                    self.__SW_Bk_Hk(**solver_input)
                            
                else:
                    solver_input['method'] = method if method != 'LA' else 'FD' # If least action method is used, first apply the full diagonalization routine
                    solver_input['mask'] = mask
                    self.__FD_ACE_Bk_Hk(**solver_input)
                
            if self.__B_k.expr.shape[0] != 0:
                # Apply the commutation relations to the operator B_k
                self.__B_k = (apply_commutation_relations(
                    self.__B_k, self.commutation_relations)).simplify()

            if order <= max_order:
                # Compute the anti-Hermitian operator S for the perturbative order
                S_k = (get_S(H0_expr, -self.__B_k, self.__do_time_dependent and not self.__is_frequency_perturbative)).simplify()
                if self.__do_time_dependent and order + self.__frequency_order <= max_order:
                    self.__dtSs[order + self.__frequency_order] = S_k.diff(t)
                # Store the anti-Hermitian operator S for the perturbative order
                self.__S[order] = S_k

            if method != 'LA':
                # Apply the commutation relations to the final Hamiltonian
                self.__Hs_final[order] = (apply_commutation_relations(self.__Hs_final.get(order, Expression()), self.commutation_relations)).simplify()
        
        if method != 'LA':
            self.__Hs_final ={
                k: (apply_substituitions(apply_commutation_relations(v, self.commutation_relations).simplify(), self.__ns)).simplify() for k, v in self.__Hs_final.items()
            }

        if method == 'LA':
            self.__Z = self.__S
            del(self.__S)
            self.__LA_solver(max_order, mask)


        # Store the results
        self.__max_order = max_order
        self.__has_mask = mask is not None

        if hasattr(self, '_EffectiveFrame__H_operator_form'):
            del (self.__H_operator_form)
        if hasattr(self, '_EffectiveFrame__H_matrix_form'):
            del (self.__H_matrix_form)
        if hasattr(self, '_EffectiveFrame__H_dict_form'):
            del (self.__H_dict_form)
        if hasattr(self, 'H'):
            del (self.H)

        logger.info('The Hamiltonian has been solved successfully. Please use the get_H method to get the result in the desired form.')

    def __prepare_result(self, O_final, return_form='operator', disable=True):
        """
        Prepares the result for the effective Hamiltonian or an operator after solving or rotating.

        This method converts the final operator expression into either operator form or matrix form depending on the
        value of the `return_form` parameter.

        Parameters
        ----------
        O_final : Expression
            The final expression for the operator or Hamiltonian after solving or rotation.
        return_form : str, optional
            If 'operator', returns the result in operator form (default is 'operator'). 
            If 'matrix', returns the matrix form.
            If 'dict' or 'dict_operator', returns the dictionary form with projected finite subspaces.
            If 'dict_matrix', returns the dictionary form with the full matrix.

        Returns
        -------
        Expr or Matrix
            The resulting operator in the chosen form. If `return_form` is True, it returns an operator expression.
            If False, it returns the matrix form of the operator.
        """

        if 'operator' in return_form:
            if self.subspaces is None and O_final.expr[0].fn.shape[0] > 1:
                logger.info('Subspaces were not provided. Creating a finite subspace with the same dimension as the Hamiltonian.')
                finite_subspace = RDBasis('f', self.H_input.shape[0])
                self.subspaces = [finite_subspace]
                self.__composite_basis = RDCompositeBasis(self.subspaces)

            if not hasattr(self, '_EffectiveFrame__composite_basis') and self.subspaces is not None:
                self.__composite_basis = RDCompositeBasis(self.subspaces)


        if return_form == 'operator':
            if self.subspaces is not None:
                O_final_projected = np_sum([self.__composite_basis.project(mul_group.fn) * Mul(
                    *mul_group.inf).simplify() for mul_group in tqdm(O_final.expr, desc='Converting to operator form', disable=disable)])

                return O_final_projected
            O_final_projected = np_sum([mul_group.fn[0] * Mul(
                *mul_group.inf).simplify() for mul_group in tqdm(O_final.expr, desc='Converting to operator form', disable=disable)])
            return O_final_projected

        elif return_form == 'matrix':
            O_matrix_form = sp_zeros(
                O_final.expr[0].fn.shape[0], O_final.expr[0].fn.shape[1])

            for mul_group in tqdm(O_final.expr, desc='Converting to matrix form', disable=disable):
                O_matrix_form += mul_group.fn * Mul(*mul_group.inf).simplify()

            if self.subspaces is None and O_final.expr[0].fn.shape[0] == 1:
                return O_matrix_form[0]

            return O_matrix_form
        
        elif 'dict' in return_form:
            return_form, extra = return_form.split('_')
            O_dict_form = {}

            if extra == 'operator':
                if self.subspaces is not None:
                    for mul_group in tqdm(O_final.expr, desc='Converting to dictionary (operator) form', disable=disable):
                        O_dict_form[Mul(
                            *mul_group.inf)] = self.__composite_basis.project(mul_group.fn)
                else:
                    for mul_group in tqdm(O_final.expr, desc='Converting to dictionary (operator) form', disable=disable):
                        O_dict_form[Mul(*mul_group.inf)] = mul_group.fn[0]
                    
            elif extra == 'matrix':
                for mul_group in tqdm(O_final.expr, desc='Converting to dictionary (matrix) form', disable=disable):
                    O_dict_form[Mul(*mul_group.inf)] = mul_group.fn.expand()
            
            else:
                raise ValueError(f'Invalid return form {return_form}. Please choose either: ' + ', '.join(
                    ['operator', 'matrix', 'dict', 'dict_operator', 'dict_matrix']))

            return O_dict_form

        raise ValueError(f'Invalid return form {return_form}. Please choose either: ' + ', '.join(
            ['operator', 'matrix', 'dict', 'dict_operator', 'dict_matrix']))

    def get_H(self, return_form=None):
        """
        Returns the effective Hamiltonian.

        Parameters
        ----------
        return_form : str, optional
            If 'operator', returns the result in operator form (default is 'operator'). 
            If 'matrix', returns the matrix form.
            If 'dict' or 'dict_operator', returns the dictionary form with projected finite subspaces.
            If 'dict_matrix', returns the dictionary form with the full matrix.

        Returns
        -------
        Expression or Matrix
            The effective Hamiltonian in the specified form.
        """

        return_form = self.__return_form if return_form is None else return_form
        self.__Hs_final = {k: v for k, v in self.__Hs_final.items() if v.expr.shape[0] != 0}

        if not hasattr(self, '_EffectiveFrame__Hs_final'):
            raise AttributeError(
                'The Hamiltonian has not been solved yet. Please run the solver method first.')

        if return_form == 'operator':
            if hasattr(self, '_EffectiveFrame__H_operator_form'):
                self.corrections = self.__H_operator_form_corrections
                self.H = self.__H_operator_form
                return self.__H_operator_form

            self.__H_operator_form_corrections = {k: self.__prepare_result(v, return_form) for k, v in tqdm(self.__Hs_final.items(), desc='Converting to operator form', disable= not self.verbose)}
            self.__H_operator_form = np_sum(list(self.__H_operator_form_corrections.values()))
            self.H = self.__H_operator_form
            self.corrections = self.__H_operator_form_corrections

        elif return_form == 'matrix':
            if hasattr(self, '_EffectiveFrame__H_matrix_form'):
                self.corrections = self.__H_matrix_form_corrections
                self.H = self.__H_matrix_form
                return self.__H_matrix_form
            
            self.__H_matrix_form_corrections = {k: self.__prepare_result(v, return_form) for k, v in tqdm(self.__Hs_final.items(), desc='Converting to matrix form', disable= not self.verbose)}
            self.__H_matrix_form = sp_Add(*list(self.__H_matrix_form_corrections.values()))
            self.H = self.__H_matrix_form
            self.corrections = self.__H_matrix_form_corrections


        elif 'dict' in return_form:
            extra = return_form.split(
                '_')[1] if '_' in return_form else self.__return_form
            if hasattr(self, '_EffectiveFrame__H_dict_form') and self.__H_dict_form.get(extra) is not None:
                self.corrections = self.__H_dict_form_corrections[extra]
                self.H = self.__H_dict_form[extra]
                return self.__H_dict_form[extra]
            
            if not hasattr(self, '_EffectiveFrame__H_dict_form'):
                self.__H_dict_form = {}
                self.__H_dict_form_corrections  = {}

            self.__H_dict_form_corrections[extra] = {k: self.__prepare_result(v, 'dict' + f'_{extra}') for k, v in tqdm(self.__Hs_final.items(), desc=f'Converting to dictionary of {extra} form', disable= not self.verbose)}

            self.__H_dict_form[extra] = {}
            
            for _, v in self.__H_dict_form_corrections[extra].items():
                for k, v1 in v.items():
                    if self.__H_dict_form.get(k):
                        self.__H_dict_form[extra][k] += v1
                    else:
                        self.__H_dict_form[extra][k] = v1
            
            self.H = self.__H_dict_form[extra]
            self.corrections = self.__H_dict_form_corrections[extra]
            
        else:
            raise ValueError('Invalid return form. Please choose either: ' + ', '.join(
                ['operator', 'matrix', 'dict', 'dict_operator', 'dict_matrix']))
        
        return self.H
    
    def __rotate(self, max_order, Os):
        factorials = [sp_Rational(1, sp_factorial(k)) for k in range(0, max_order + 1)]
        nest_commute = create_nest_commute([Os, self.__dtSs], self.__S)

        result = {}
        result[0] = Os.get(0, Expression()) + Expression()

        for order in trange(1, max_order + 1, desc='Rotating for each order', disable=not self.verbose):
            set_of_keys = partitions(order)
            # Iterate over the all the partitions. Do not eliminate any partition.
            for key in set_of_keys:
                if len(key) == 1:
                    result[order] = apply_commutation_relations(result.get(order, Expression()) + Os.get(key[0], Expression()), self.commutation_relations).simplify()
                    continue

                nestedness = len(key) - 1
                result[order] = apply_commutation_relations(result.get(order, Expression()) + nest_commute(key, 0) * factorials[nestedness], self.commutation_relations).simplify()
                if self.__do_time_dependent:
                    result[order] = apply_commutation_relations(result.get(order, Expression()) - I * hbar * nest_commute(key, 1) * factorials[nestedness + 1], self.commutation_relations).simplify()
        
        return result

    def rotate(self, expr, max_order=None, return_form=None):
        """
        Rotates a given expression according to the computed transformation S.

        Parameters
        ----------
        expr : Expression
            The expression to rotate.
        max_order : int, optional
            The maximum order to consider during the rotation (default is None).
        return_form : str, optional
            If 'operator', returns the result in operator form (default is 'operator').
            If 'matrix', returns the matrix form.
            If 'dict' or 'dict_operator', returns the dictionary form with projected finite subspaces.
            If 'dict_matrix', returns the dictionary form with the full matrix.

        Returns
        -------
        Expression or Matrix
            The rotated expression in the specified form.
        """
        if max_order is None:
            max_order = max(self.__S.keys())

        return_form = self.__return_form if return_form is None else return_form

        Os = get_perturbative_expression(
            expr, self.__structure, self.subspaces)

        result_dict = self.__rotate(max_order, Os)
        result = np_sum(list(result_dict.values()))

        result = (apply_commutation_relations(
            result, self.commutation_relations)).simplify()
        
        if 'dict' in return_form:
            extra = return_form.split(
                '_')[1] if '_' in return_form else self.__return_form
            return_form = 'dict' + f'_{extra}'

        return self.__prepare_result(result, return_form, disable=False)

    def __str__(self):
        information = '\nEffective Frame\n\n'

        subspaces_headers = [['Name', 'Type', 'Dimension']]

        subspaces_finite = [[subspace.name, 'Finite' , f'{subspace.dim}x{subspace.dim}']
                            for subspace in self.subspaces if subspace.name != 'finite_pysw_built_in_function'] if self.subspaces is not None else []
        
        if self.__return_form == 'matrix' and self.subspaces is None:
            subspaces_finite = [['Finite', 'Finite', f'{self.H_input.shape[0]}x{self.H_input.shape[0]}']]

        subspaces_infinite = [[subspace.as_ordered_factors(
        )[1].name, 'Bosonic', '∞'] for subspace in self.__structure.keys()]

        subspaces_info = tabulate(subspaces_headers + subspaces_finite + subspaces_infinite,
                                  headers='firstrow', tablefmt='rounded_grid', stralign='center')

        information += subspaces_info

        information += '\n\nEffective Hamiltonian: '
        if not hasattr(self, '_EffectiveFrame__H_final'):
            information += '\tNot computed yet. To do so, run `solve` method. '
        else:
            information += f'\tComputed to {self.__max_order} order using'
            information += ' full diagonalization routine.' if self.__full_diagonalization else ' mask routine.' if self.__has_mask else ' regular Schrieffer-Wolff transformation.'

        information += '\n\n'

        return information
    
    def _repr_latex_(self):
        latex_str = r'\text{Effective Frame}\\'

        # Header for subspaces
        subspaces_headers = [r'\mathrm{Name}', r'\mathrm{Type}', r'\mathrm{Dimension}']

        # Finite subspaces
        subspaces_finite = [[sp_latex(RDSymbol(subspace.name)), r'\text{Finite}', f'{subspace.dim}\\times{subspace.dim}']
                            for subspace in self.subspaces if subspace.name != 'finite_pysw_built_in_function'] if self.subspaces is not None else []

        if self.__return_form == 'matrix' and self.subspaces is None:
            subspaces_finite = [['Finite', 'Finite', f'{self.H_input.shape[0]}x{self.H_input.shape[0]}']]

        # Infinite subspaces
        subspaces_infinite = [[sp_latex(RDSymbol(str(subspace.as_ordered_factors(
        )[1].name))), r'\text{Bosonic}', r'\infty'] for subspace in self.__structure.keys()]

        # Combine all subspaces
        subspaces_table = subspaces_finite + subspaces_infinite

        # Create LaTeX table
        latex_table = r'\begin{array}{ccc}\hline ' + ' & '.join(subspaces_headers) + r' \\ \hline '
        for row in subspaces_table:
            latex_table += ' & '.join(row) + r' \\ '
        latex_table += r'\hline \end{array} \\'

        latex_str +=  latex_table

        # Effective Hamiltonian information
        latex_str += r'\text{Effective Hamiltonian: }'
        if not hasattr(self, '_EffectiveFrame__H_final'):
            latex_str += r'\text{Not computed yet.}\\ \text{To do so, run \texttt{solve} method.}'
        else:
            latex_str += f'\\text{{Computed to {self.__max_order} order using }}'
            if self.__full_diagonalization:
                latex_str += r'\text{full diagonalization routine.}'
            elif self.__has_mask:
                latex_str += r'\text{mask routine.}'
            else:
                latex_str += r'\text{regular Schrieffer-Wolff transformation.}'

        return  '$' + latex_str + '$'
