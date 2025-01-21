
"""
Branch: Optimizations branch
Title: Solver for sympt package
Date: 8 January 2025
Authors:
- Giovanni Francesco Diotallevi
- Irving Leander Reascos Valencia

DOI: https://doi.org/10.48550/arXiv.2412.10240


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

[ ] Eta (W) functions can be optimized by using the Hermitian property of Ps and the anti-Hermitian (Hermitian) property of Eta (W).
[ ] Implement parallelization for the computation of the anti-Hermitian Operator Eta (W).

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




def get_Eta(W_delta, P_, correct_denominator=False):

    if not np_any(P_.expr):
        return Expression()
    
    Eta = Expression()

    for mul_group in P_.expr:
        Delta = mul_group.delta
        P_mat = MutableDenseMatrix(mul_group.fn)
        P_mat_non_zeros = np_nonzero(P_mat)

        D = W_delta(tuple(Delta))
        
        if not correct_denominator:
            if not np_any(D[P_mat_non_zeros]):
                raise ValueError('The energy denominator is zero for the given matrix elements.')

            D[D == 0] = 1
            Eta = Eta + MulGroup(sp_elementwise(sp_Matrix(1/D), P_mat), mul_group.inf, mul_group.delta, mul_group.Ns)
            continue

        Eta_mat = sp_zeros(*P_mat.shape)

        for i, j in zip(*P_mat_non_zeros):
            if i == j and not np_any(Delta):
                raise ValueError('It is not possible to eliminate diagonal elements.')
            corrected_eta = 0
            for term in P_mat[i, j].expand().as_ordered_terms():
                
                if not term.has(t):
                    corrected_eta += term / D[i, j]
                    continue

                freq_term = (extract_frequencies(term) * (-I  * hbar)).cancel()  # We assume that the argument for the exponential is in the form of exp(i/hbar Energy t)
                denominator = D[i, j] + freq_term
                if denominator == 0:
                    raise ValueError(f'It is impossible to decouple the state with index {int(i)} (Delta {Delta}) from state with index {int(j)} (Delta {Delta}) because they are degenerate (see H0)')
                corrected_eta += term / denominator
            Eta_mat[i, j] = corrected_eta
        
        Eta = Eta + MulGroup(Eta_mat, mul_group.inf, mul_group.delta, mul_group.Ns)
            
    return Eta


def w_uv_delta(H0_expr, uu, vv, Delta):
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

def get_w_delta(H0_expr):
    fin_shape = H0_expr.expr[0].fn.shape

    def w_delta(delta):
        delta = np_array(delta)
        res = np_zeros(fin_shape, dtype=object)
        for uu in range(fin_shape[0]):
            for vv in range(fin_shape[1]):
                if np_all(delta == 0) and uu == vv:
                    continue
                res[uu, vv] = w_uv_delta(H0_expr, uu, vv, delta)
        return res

    return memoized(w_delta)


def get_P_generator(Hs_memory, Vs_memory, Ws_memory, Etas_memory, Ps_memory):
    # Hs contains only the B(H) terms
    # Vs contains only the B_(H) terms 

    @memoized
    def W(i, j):
        if i not in Hs_memory and i not in Vs_memory:
            return Expression(), Expression()
        W_to_remove, W_to_keep = Ws_memory[j]

        Wij_to_keep   = commutator(Vs_memory.get(i, Expression()), W_to_remove) + commutator(Hs_memory.get(i, Expression()), W_to_keep)

        Wij_to_remove = commutator(Vs_memory.get(i, Expression()), W_to_keep) + commutator(Hs_memory.get(i, Expression()), W_to_remove)


        return Wij_to_remove, Wij_to_keep
    
    @memoized
    def E(i, j):
        if i == 0:
            return Ps_memory[j][0], Expression()
        if i not in Hs_memory and i not in Vs_memory:
            return Expression(), Expression()
        
        Eij_to_keep   = commutator(Vs_memory.get(i, Expression()), Etas_memory[j])
        Eij_to_remove = commutator(Hs_memory.get(i, Expression()), Etas_memory[j])

        return Eij_to_remove, Eij_to_keep
    
    @memoized
    def WE_we(i, j, k):
        W_to_remove, W_to_keep = W(j, i)
        E_to_remove, E_to_keep = E(j, i)

        Wk_to_remove, Wk_to_keep = Ws_memory[k]

        # Ww = (Wk + WR) (wk + wR)
        term_to_keep   = W_to_keep * Wk_to_keep   + W_to_remove * Wk_to_remove
        term_to_remove = W_to_keep * Wk_to_remove + W_to_remove * Wk_to_keep
        # -W eta = - (Wk + WR) eta
        term_to_keep = term_to_keep     - W_to_remove * Etas_memory[k]
        term_to_remove = term_to_remove - W_to_keep   * Etas_memory[k]
        # Ew = (Ek + ER) (wk + wR)
        term_to_keep = term_to_keep + E_to_keep * Wk_to_keep + E_to_remove * Wk_to_remove
        term_to_remove = term_to_remove + E_to_keep * Wk_to_remove + E_to_remove * Wk_to_keep
        # -E eta = - (Ek + ER) eta
        term_to_keep = term_to_keep     - E_to_remove * Etas_memory[k]
        term_to_remove = term_to_remove - E_to_keep   * Etas_memory[k]


        term_to_keep = term_to_keep + term_to_keep.dagger()
        term_to_remove = term_to_remove + term_to_remove.dagger()

        #term = (W(j, i) + E(j, i))  * (Ws_memory[k] - Etas_memory[k])
        #term = term + term.dagger()
        return term_to_remove, term_to_keep
        
        
    def P_uu_vv_delta(order):

        P_to_keep   = Hs_memory.get(order, Expression())
        P_to_remove = Vs_memory.get(order, Expression())

        for i, j in T(order, 2):
            Eij_to_remove, Eij_to_keep = E(i, j)

            P_to_remove = P_to_remove - Eij_to_remove
            P_to_keep = P_to_keep - Eij_to_keep
    
        for i, j, k in T(order, 3):
            WE_we_ikl_to_remove, WE_we_ikl_to_keep = WE_we(i, j, k)
            P_to_remove = P_to_remove - sp_Rational(1, 2) * WE_we_ikl_to_remove
            P_to_keep =   P_to_keep   - sp_Rational(1, 2) * WE_we_ikl_to_keep

        for i, j in T(order, 2):
            WE_we_i_j_to_remove, WE_we_i_j_to_keep = WE_we(i, 0, j)
            P_to_remove = P_to_remove - sp_Rational(1, 2) * WE_we_i_j_to_remove
            P_to_keep =   P_to_keep   - sp_Rational(1, 2) * WE_we_i_j_to_keep
        
        return P_to_remove, P_to_keep
    
    return memoized(P_uu_vv_delta)


def get_W_generator(Ws_memory, Eta_memory):

    def get_W(order):
        W_to_remove, W_to_keep = Expression(), Expression()
        for i, j in T(order, 2):
            Wi_to_remove, Wi_to_keep = Ws_memory.get(i, (Expression(), Expression()))
            Wj_to_remove, Wj_to_keep = Ws_memory.get(j, (Expression(), Expression()))
            Ei = Eta_memory.get(i, Expression())
            Ej = Eta_memory.get(j, Expression())

            # (Wi_k + Wi_R) * (Wj_k + Wj_R)
            W_to_keep   = W_to_keep   + Wi_to_keep * Wj_to_keep + Wi_to_remove * Wj_to_remove
            W_to_remove = W_to_remove + Wi_to_keep * Wj_to_remove + Wi_to_remove * Wj_to_keep

            # -(Wi_k + Wi_R) * Ej
            #W_to_keep   = W_to_keep   - Wi_to_remove * Ej
            #W_to_remove = W_to_remove - Wi_to_keep   * Ej
            # Ei * (Wj_k + Wj_R)
            #W_to_keep   = W_to_keep   + Ei * Wj_to_remove
            #W_to_remove = W_to_remove + Ei * Wj_to_keep

            # - Ei * Ej
            W_to_keep   = W_to_keep   - Ei * Ej
            

        return  W_to_remove * (- sp_Rational(1,2)) , W_to_keep * (- sp_Rational(1,2))
    
    return memoized(get_W)



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
        if method not in ['SW', 'FD', 'ACE']:
            raise ValueError('Invalid method. Please choose one of the following: SW, FD, ACE.')
        

        self.__H_old = copy(self.H_input)
        self.__V_old = copy(self.V_input)

        # Check if the perturbative interaction is provided
        if method == 'SW':
            if self.V_input is None:
                # Raise an error if the perturbative interaction is not provided
                raise ValueError(
                    'The perturbative interaction must be provided for the regular Schrieffer-Wolff transformation')
            b_p = lambda ps : ps

        # Check if the mask is used in the full diagonalization mode
        if method == 'FD':
            if mask is not None:
                mask = None
                # If the mask is used in the full diagonalization mode, it will be ignored
                logger.info(
                    'The mask is not used in the full diagonalization mode and it will be ignored')
            
            b_p = lambda x: separate_diagonal_off_diagonal(x[0] + x[1])


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

            b_p = lambda x: mask.apply_mask(x[0] + x[1])

        elif method == 'ACE' and mask is None:
            raise ValueError(f'The mask must be provided for the {method} method.')
          
        # Compute the perturbative expression for the Hamiltonian
        Hs_total = get_perturbative_expression(
            self.__H_old, self.__structure, self.subspaces)
        
        # If the full diagonalization or mask routine is used
        self.__Vs = {}
        self.__Hs = {}

        # If the regular Schrieffer-Wolff transformation is used
        if method == 'SW':
            # Set the Hamiltonians to the perturbative expressions
            self.__Hs = Hs_total
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
            # Iterate over the perturbative expressions for the Hamiltonian
            for h_order, h_expr in Hs_total.items():
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

                new_vk, new_hk = b_p((h_expr, Expression()))

                self.__Hs[h_order] = new_hk
                self.__Vs[h_order] = new_vk

        # Apply the commutation relations to the zeroth-order Hamiltonian
        H0_expr = apply_commutation_relations(
        self.__Hs.get(0) + Expression(), self.commutation_relations).simplify()
        if H0_expr.is_time_dependent:
            raise ValueError('The zeroth order of the Hamiltonian must be time-independent.')
        self.__do_time_dependent = np_any([v.is_time_dependent for k, v in self.__Hs.items() if k != 0]) or np_any([v.is_time_dependent for k, v in self.__Vs.items() if k != 0])

        if self.__do_time_dependent:

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
            
            H0_expr = self.__Hs.get(0)
        
        # Extract the number operators from  the zeroth-order Hamiltonian
        H0_expr, self.__ns = extract_ns(H0_expr, self.__structure)
        H0_expr = H0_expr.simplify()

        # Initialize the dictionary to store the anti-Hermitian operator S for each order
        self.__Up = {0: Expression()}
        self.__Upc = {0: Expression()}
        self.__Ws = {0: (Expression(), Expression()), 1: (Expression(), Expression())}
        self.__dtWs = {0: (Expression(), Expression()), 1: (Expression(), Expression())}
        self.__Etas = {0: Expression()}
        self.__dtEtas = {0: Expression()}
        self.__Ps = {0: (Expression(), Expression())}
        self.__Qs = {0: Expression()}
        self.__Hs_final = {0 : H0_expr}

        def B_P(order):
            return b_p(self.__Ps[order])

        return H0_expr, B_P
    
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
                - ACE : Arbitrary coupling elimination least action.
        mask : Expression, optional
            A mask expression used for selectively applying transformations (default is None).
        """

        # Prepare the Hamiltonians and perturbative interactions
        H0_expr, B_P = self.__checks_and_prepare_solver(method, mask, max_order)

        Ps = get_P_generator(self.__Hs, self.__Vs, self.__Ws, self.__Etas, self.__Ps)
        W_delta = get_w_delta(H0_expr)
        get_W = get_W_generator(self.__Ws, self.__Etas)

        for order in trange(1, max_order + 1, desc='Computing the effective Hamiltonian', disable=not self.verbose):
            self.__Ps[order] = Ps(order)

            if self.__do_time_dependent:
                Q_order_to_remove, Q_order_to_keep = Expression(), Expression()

                #dtEta[order]
                self.__Ps[order] = (self.__Ps[order][0] + I * hbar * self.__dtEtas.get(order, Expression()), self.__Ps[order][1])

                for i, j in T(order, 2):

                    dtWi_to_remove, dtWi_to_keep = self.__dtWs.get(i, (Expression(), Expression()))
                    Wj_to_remove, Wj_to_keep = self.__Ws.get(j, (Expression(), Expression()))

                    # (dtWi_k + dtWi_R) * (Wj_k + Wj_R)
                    Q_order_to_keep   = Q_order_to_keep   + dtWi_to_keep * Wj_to_keep + dtWi_to_remove * Wj_to_remove
                    Q_order_to_remove = Q_order_to_remove + dtWi_to_keep * Wj_to_remove + dtWi_to_remove * Wj_to_keep

                    # -(dtWi_k + dtWi_R) * Ej
                    Q_order_to_keep   = Q_order_to_keep   - dtWi_to_remove * self.__Etas[j]
                    Q_order_to_remove = Q_order_to_remove - dtWi_to_keep   * self.__Etas[j]

                    # dtEi * (Wj_k + Wj_R)
                    Q_order_to_keep   = Q_order_to_keep   + self.__dtEtas.get(i, Expression()) * Wj_to_remove
                    Q_order_to_remove = Q_order_to_remove + self.__dtEtas.get(i, Expression()) * Wj_to_keep

                    # - dtEi * Ej
                    Q_order_to_keep   = Q_order_to_keep   - self.__dtEtas.get(i, Expression()) * self.__Etas[j]

                Q_order_to_keep   = (Q_order_to_keep   - Q_order_to_keep.dagger())   * I * hbar * sp_Rational(1,2)
                Q_order_to_remove = (Q_order_to_remove - Q_order_to_remove.dagger()) * I * hbar * sp_Rational(1,2)
                
                self.__Qs[order] = (Q_order_to_remove, Q_order_to_keep)
                self.__Ps[order] = (self.__Ps[order][0] + Q_order_to_remove, self.__Ps[order][1] + Q_order_to_keep)

            self.__Ps[order] = [apply_commutation_relations(P_t, self.commutation_relations).simplify() for P_t in self.__Ps[order]]

            P_to_remove, P_to_keep = B_P(order)
            self.__Ps[order] = [P_to_remove, P_to_keep]
            self.__Hs_final[order] = P_to_keep
            self.__Etas[order] = get_Eta(W_delta, P_to_remove, correct_denominator = self.__do_time_dependent and not self.__is_frequency_perturbative)
            self.__Ws[order] = [apply_commutation_relations(W_t, self.commutation_relations).simplify() for W_t in get_W(order)]
            self.__Up[order] = np_sum(self.__Ws.get(order, (Expression(), Expression()))) + self.__Etas[order]
            self.__Upc[order] = np_sum(self.__Ws.get(order, (Expression(), Expression()))) - self.__Etas[order]

            perturbative_order = order + self.__frequency_order if self.__do_time_dependent else order

            if self.__do_time_dependent and perturbative_order <= max_order:
                self.__dtWs[perturbative_order]   = [W_t.diff(t) for W_t in self.__Ws[order]]
                self.__dtEtas[perturbative_order] = self.__Etas[order].diff(t)
                if perturbative_order == order:
                    self.__Ps[order] = (self.__Ps[order][0] + I * hbar * self.__dtEtas.get(order, Expression()), self.__Ps[order][1])

        self.__Hs_final ={
                k: (apply_substituitions(v, self.__ns)).simplify() for k, v in self.__Hs_final.items()
            }

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
                O_final_projected = np_sum([np_sum([v.cancel() * k for k,v in group_by_operators(self.__composite_basis.project(mul_group.fn)).items()]) * Mul(
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
                            *mul_group.inf)] = np_sum([v.cancel() * k for k,v in group_by_operators(self.__composite_basis.project(mul_group.fn)).items()])
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
            max_order = self.__max_order

        return_form = self.__return_form if return_form is None else return_form

        Os = get_perturbative_expression(
            expr, self.__structure, self.subspaces)
        
        H_rotated = {0: Os.get(0, Expression())}

        for order in trange(1, max_order + 1, desc='Rotating the expression', disable= not self.verbose):
            H_rotated_order = [Os.get(order, Expression()), (self.__Up[order] * Os.get(0, Expression())), (Os.get(0, Expression()) * self.__Upc[order])]

            for i, j in list(T(order, 2)):   
                H_rotated_order.append((self.__Up[i] * Os.get(0, Expression()) * self.__Upc[j]))
                H_rotated_order.append((self.__Up[i] * Os.get(j, Expression())))
                H_rotated_order.append((Os.get(i, Expression()) * self.__Upc[j]))
            
            for i,j,k in list(T(order, 3)):
                H_rotated_order.append((self.__Up[i] * Os.get(j, Expression()) * self.__Upc[k]))

            if self.__do_time_dependent:
                H_rotated_order.append(I * hbar * np_sum(self.__Qs[order]))

            H_rotated[order] = apply_substituitions(apply_commutation_relations(np_sum(H_rotated_order), self.commutation_relations), self.__ns).simplify()
                

        result = np_sum(list(H_rotated.values())).simplify()
        
        if 'dict' in return_form:
            extra = return_form.split(
                '_')[1] if '_' in return_form else self.__return_form
            return_form = 'dict' + f'_{extra}'

        return self.__prepare_result(result, return_form, disable=False)
