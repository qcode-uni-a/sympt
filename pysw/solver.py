
"""
Title: Solver for PySW package
Date: 17 October 2024
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
                   UnevaluatedExpr)
# import deep copy
from copy import copy, deepcopy


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


def get_S(H0_expr, equation_to_solve):
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
            if denom == 0:
                if uu == vv and Delta == 0:
                    raise ValueError(f'S contains diagonal elements. If you saw this message, please contact the developers.')
                raise ValueError(f'You are trying to decouple degenarate states for the finite entry {uu, vv} and Delta {Delta}')
            S_mat[uu, vv] /= denom
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

    def __init__(self, H, V=None, subspaces=None):
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
        self.H_input = H
        self.V_input = V

        self.subspaces = subspaces
        self.__return_form = 'operator'
        self.__structure = count_bosonic_subspaces(self.H_input)
        self.commutation_relations = self.get_commutation_relations()

        self.formatter()

        print('The EffectiveFrame object has been initialized successfully.')
        print(self.__str__())

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
            print('Creating the EffectiveFrame object with matrix form.')
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

    def __checks_solver(self, do_regular_SW, full_diagonalization, mask):

        self.__H_old = copy(self.H_input)
        self.__V_old = copy(self.V_input)

        # Check if the perturbative interaction is provided
        if do_regular_SW and self.V_input is None:
            # Raise an error if the perturbative interaction is not provided
            raise ValueError(
                'The perturbative interaction must be provided for the regular Schrieffer-Wolff transformation')

        # Check if the mask is used in the full diagonalization mode
        if full_diagonalization and mask is not None:
            mask = None
            # If the mask is used in the full diagonalization mode, it will be ignored
            print(
                'The mask is not used in the full diagonalization mode and it will be ignored')

        # Check if the perturbative interaction will be added to the full Hamiltonian
        if self.V_input is not None and not do_regular_SW:
            print('The perturbative interaction will be added to the full Hamiltonian')
            # Add the perturbative interaction to the Hamiltonian
            self.__H_old = (self.H_input + self.V_input).expand()
            # Set the perturbative interaction to zero
            self.__V_old = S.Zero

        # Add the Hermitian conjugate and the blocks to the mask. This ensures that the mask is Hermitian
        mask = mask + mask.hermitian() + Blocks() if mask is not None else None
        return mask

    def __prepare_Hs_Vs(self, do_regular_SW, mask):
        # Compute the perturbative expression for the Hamiltonian
        Hs_aux = get_perturbative_expression(
            self.__H_old, self.__structure, self.subspaces)

        # If the regular Schrieffer-Wolff transformation is used
        if do_regular_SW:
            # Set the Hamiltonians to the perturbative expressions
            Hs = Hs_aux
            # Compute the perturbative expression for the perturbative interaction
            Vs = get_perturbative_expression(
                self.__V_old, self.__structure, self.subspaces)
            if Vs.get(0) is not None:
                # Check if the zeroth order of the perturbative interaction is zero
                raise ValueError(
                    f'The zeroth order of the perturbative interaction is not zero, but it is instead: {Vs.get(0)}')
            # Apply the commutation relations to the zeroth-order Hamiltonian
            H0_expr = apply_commutation_relations(
            Hs.get(0) + Expression(), self.commutation_relations).simplify()
            return Hs, Vs, H0_expr

        # If the full diagonalization or mask routine is used
        Vs = {}
        Hs = {}
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
                Hs[h_order] = new_hk
                continue

            # If the mask is used
            if mask is not None:
                # Apply the mask to the perturbative expression
                new_vk, new_hk = mask.apply_mask(h_expr)
            # If the full diagonalization routine is used
            else:
                # Separate the diagonal and off-diagonal terms of the Hamiltonian
                new_vk, new_hk = separate_diagonal_off_diagonal(h_expr)

            Hs[h_order] = new_hk
            Vs[h_order] = new_vk

        # Apply the commutation relations to the zeroth-order Hamiltonian
        H0_expr = apply_commutation_relations(
        Hs.get(0) + Expression(), self.commutation_relations).simplify()
        return Hs, Vs, H0_expr

    def solve(self, max_order=2, full_diagonalization=False, mask=None):
        """
        Solves for the effective Hamiltonian up to the specified order using the Schrieffer-Wolff transformation.

        Parameters
        ----------
        max_order : int, optional
            The maximum perturbative order to solve for (default is 2).
        full_diagonalization : bool, optional
            Whether to perform a full diagonalization of the Hamiltonian (default is False).
        mask : Expression, optional
            A mask expression used for selectively applying transformations (default is None).
        """
        do_regular_SW = not full_diagonalization and mask is None                                                              # Check if the regular Schrieffer-Wolff transformation should be used or not
        mask = self.__checks_solver(do_regular_SW, full_diagonalization, mask)

        if mask is not None:
            # Add the structure of the bosonic subspaces to the mask
            mask.add_structure(self.__structure)

        # Prepare the Hamiltonians and perturbative interactions
        Hs, Vs, H0_expr = self.__prepare_Hs_Vs(do_regular_SW, mask)
        # Compute the factorials for the perturbative orders
        factorials = [sp_Rational(1, sp_factorial(k))
                      for k in range(0, max_order + 1)]
        # Initialize the dictionary to store the anti-Hermitian operator S for each order
        Ss = {}

        @memoized
        def nest_commute(parts, is_Vs):
            '''
            This function computes the nested commutator for the Schrieffer-Wolff transformation.

            Parameters
            ----------
            parts : list
                The list of parts to compute the nested commutator.
            is_Vs : bool
                A boolean value indicating whether the nested commutator is for the perturbative interaction.
                If True, the nested commutator is for the perturbative interaction; otherwise, it is for the block-diagonal part.
            '''
            if len(parts) == 2:
                Os = Vs if is_Vs else Hs
                return commutator(Os.get(parts[0], 0), Ss[parts[1]])
            return commutator(nest_commute(parts[:-1], is_Vs), Ss[parts[-1]])

        # Extract the number operators from  the zeroth-order Hamiltonian
        H0_expr, ns_comms = extract_ns(H0_expr, self.__structure)
        H0_expr = H0_expr.simplify()
        H_final = H0_expr

        # Iterate over the perturbative orders
        for order in trange(1, max_order + 1, desc='Solving for each order'):

            # Compute the partitions for the perturbative order
            set_of_keys = partitions(order)
            # Initialize the operator B_k for the perturbative order
            B_k = Expression()

            # Iterate over the partitions. Eliminate the last partition because it is the term [H0, S] and it is used to compute the operator S
            for key in set_of_keys[:-1]:

                if len(key) == 1:
                    # Does not deppend of full_diagonalization neither on mask
                    Vk = Vs.get(key[0], Expression())
                    # Add the perturbative interaction to the operator B_k (Equation to solve)
                    B_k += Vk
                    # Add the perturbative Hamiltonian to the final Hamiltonian
                    H_final += Hs.get(key[0], Expression())
                    continue

                # Compute the nestedness of the partition
                nestedness = len(key) - 1

                # If the regular Schrieffer-Wolff transformation is used
                if do_regular_SW:
                    # Compute the nested commutator for the regular Schrieffer-Wolff transformation
                    B_k = (B_k + nest_commute(key, nestedness %
                           2 == 0) * factorials[nestedness]).simplify()
                    # Compute the nested commutator for the regular Schrieffer-Wolff transformation
                    H_final = (H_final + nest_commute(key, nestedness %
                               2 != 0) * factorials[nestedness]).simplify()
                # If the full diagonalization or mask routine is used
                else:
                    # Compute the nested commutator for the full diagonalization or mask routine
                    new_commutator_odd = nest_commute(
                        key, nestedness % 2 != 0) * factorials[nestedness]
                    # Compute the nested commutator for the full diagonalization or mask routine
                    new_commutator_even = nest_commute(
                        key, nestedness % 2 == 0) * factorials[nestedness]
                    # Compute the nested commutator for the full diagonalization or mask routine
                    new_commutator = (new_commutator_odd +
                                      new_commutator_even).simplify()
                    if mask is not None:                                                           # If the mask is used
                        # Apply the mask to the nested commutator
                        new_bk, new_hf = mask.apply_mask(new_commutator)
                    # If the full diagonalization routine is used
                    elif full_diagonalization:
                        # Separate the diagonal and off-diagonal terms of the nested commutator
                        new_bk, new_hf = separate_diagonal_off_diagonal(
                            new_commutator)
                    # Add the nested commutator to the operator B_k
                    B_k = (B_k + new_bk).simplify()
                    # Add the nested commutator to the final Hamiltonian
                    H_final = (H_final + new_hf).simplify()

            if B_k.expr.shape[0] != 0:
                # Apply the commutation relations to the operator B_k
                B_k = (apply_commutation_relations(
                    B_k, self.commutation_relations)).simplify()

            if order < max_order:
                # Compute the anti-Hermitian operator S for the perturbative order
                S_k = (get_S(H0_expr, -B_k)).simplify()
                # Store the anti-Hermitian operator S for the perturbative order
                Ss[order] = S_k
            # Apply the commutation relations to the final Hamiltonian
            H_final = (apply_commutation_relations(
                H_final, self.commutation_relations)).simplify()

        H_final = (apply_substituitions(apply_commutation_relations(H_final, self.commutation_relations).simplify(
            # Apply the commutation relations to the final Hamiltonian
        ), ns_comms)).simplify()

        # Store the results
        self.__max_order = max_order
        self.__S = Ss
        self.__H_final = H_final
        self.__full_diagonalization = full_diagonalization
        self.__has_mask = mask is not None

        if hasattr(self, '_EffectiveFrame__H_operator_form'):
            del (self.__H_operator_form)
        if hasattr(self, '_EffectiveFrame__H_matrix_form'):
            del (self.__H_matrix_form)
        if hasattr(self, '_EffectiveFrame__H_dict_form'):
            del (self.__H_dict_form)
        if hasattr(self, 'H'):
            del (self.H)

        print('The Hamiltonian has been solved successfully. Please use the get_H method to get the result in the desired form.')

    def __prepare_result(self, O_final, return_form='operator'):
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
                print('Subspaces were not provided. Creating a finite subspace with the same dimension as the Hamiltonian.')
                finite_subspace = RDBasis('f', self.H_input.shape[0])
                self.subspaces = [finite_subspace]
                self.__composite_basis = RDCompositeBasis(self.subspaces)

            if not hasattr(self, '_EffectiveFrame__composite_basis') and self.subspaces is not None:
                self.__composite_basis = RDCompositeBasis(self.subspaces)


        if return_form == 'operator':
            if self.subspaces is not None:
                O_final_projected = np_sum([self.__composite_basis.project(mul_group.fn) * Mul(
                    *mul_group.inf).simplify() for mul_group in tqdm(O_final.expr, desc='Projecting to operator form')])

                return O_final_projected
            O_final_projected = np_sum([mul_group.fn[0] * Mul(
                *mul_group.inf).simplify() for mul_group in tqdm(O_final.expr, desc='Projecting to operator form')])
            return O_final_projected

        elif return_form == 'matrix':
            O_matrix_form = sp_zeros(
                O_final.expr[0].fn.shape[0], O_final.expr[0].fn.shape[1])

            for mul_group in tqdm(O_final.expr, desc='Converting to matrix form'):
                O_matrix_form += mul_group.fn * Mul(*mul_group.inf).simplify()

            if self.subspaces is None:
                return O_matrix_form[0]

            return O_matrix_form
        
        elif 'dict' in return_form:
            return_form, extra = return_form.split('_')
            O_dict_form = {}

            if extra == 'operator':
                if self.subspaces is not None:
                    for mul_group in tqdm(O_final.expr, desc='Converting to dictionary (operator) form'):
                        O_dict_form[Mul(
                            *mul_group.inf)] = self.__composite_basis.project(mul_group.fn)
                else:
                    for mul_group in tqdm(O_final.expr, desc='Converting to dictionary (operator) form'):
                        O_dict_form[Mul(*mul_group.inf)] = mul_group.fn[0]
                    
            elif extra == 'matrix':
                for mul_group in tqdm(O_final.expr, desc='Converting to dictionary (matrix) form'):
                    O_dict_form[Mul(*mul_group.inf)] = mul_group.fn.expand()

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

        if not hasattr(self, '_EffectiveFrame__H_final'):
            raise AttributeError(
                'The Hamiltonian has not been solved yet. Please run the solver method first.')

        if return_form == 'operator':
            if hasattr(self, '_EffectiveFrame__H_operator_form'):
                return self.__H_operator_form

            self.__H_operator_form = self.__prepare_result(
                self.__H_final, return_form)
            self.H = self.__H_operator_form

        elif return_form == 'matrix':
            if hasattr(self, '_EffectiveFrame__H_matrix_form'):
                return self.__H_matrix_form

            self.__H_matrix_form = self.__prepare_result(
                self.__H_final, return_form)
            self.H = self.__H_matrix_form

        elif 'dict' in return_form:
            extra = return_form.split(
                '_')[1] if '_' in return_form else self.__return_form
            if hasattr(self, '_EffectiveFrame__H_dict_form') and self.__H_dict_form.get(extra) is not None:
                return self.__H_dict_form[extra]

            self.__H_dict_form = {}
            self.__H_dict_form[extra] = self.__prepare_result(
                self.__H_final, 'dict'+f'_{extra}')
            self.H = self.__H_dict_form[extra]
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
            max_order = max(self.__S.keys())

        return_form = self.__return_form if return_form is None else return_form

        Os = get_perturbative_expression(
            expr, self.__structure, self.subspaces)

        @memoized
        def nest_commute(parts):
            if len(parts) == 2:
                return commutator(Os.get(parts[0], 0), self.__S[parts[1]])
            return commutator(nest_commute(parts[:-1]), self.__S[parts[-1]])

        factorials = [sp_Rational(1, sp_factorial(k))
                      for k in range(0, max_order + 1)]
        result = Os.get(0, Expression()) + Expression()

        for order in trange(1, max_order + 1, desc='Rotating for each order'):
            set_of_keys = partitions(order)
            # Iterate over the all the partitions. Do not eliminate any partition.
            for key in set_of_keys:
                if len(key) == 1:
                    result += Os.get(key[0], Expression())
                    continue

                nestedness = len(key) - 1
                result += (nest_commute(key) *
                           factorials[nestedness]).simplify()

        result = (apply_commutation_relations(
            result, self.commutation_relations)).simplify()
        
        if 'dict' in return_form:
            extra = return_form.split(
                '_')[1] if '_' in return_form else self.__return_form
            return_form = 'dict' + f'_{extra}'

        return self.__prepare_result(result, return_form)

    def __str__(self):
        information = '\nEffective Frame\n\n'

        subspaces_headers = [['Name', 'Type', 'Dimension']]

        subspaces_finite = [[subspace.name, 'Finite', f'{subspace.dim}x{subspace.dim}']
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
