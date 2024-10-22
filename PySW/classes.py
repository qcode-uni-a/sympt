"""
Title: Classes for PySW package
Date: 17 October 2024
Authors:
- Giovanni Francesco Diotallevi
- Irving Leander Reascos Valencia

DOI: doi.doi.doi

Description:

This module defines various mathematical and quantum operator classes for use within the PySW package.
The core classes provide functionality to manipulate symbolic expressions, matrix operators, and basis
representations used in quantum mechanics or related fields. The module leverages SymPy and NumPy to 
handle symbolic mathematics and numerical operations, offering a foundation for more complex operations 
like tensor products, projections, and algebraic manipulation of operators.

Classes:
--------
- `Blocks`: Represents a collection of `Block` objects with methods to simplify, add, and apply masks.
- `Block`: Represents a matrix block with certain algebraic structures.
- `Expression`: Represents a symbolic or numeric expression using NumPy arrays.
- `MulGroup`: Represents a multiplicative group for symbolic matrix multiplication.
- `RDOperator`: Represents a quantum operator with a matrix and associated subspace.
- `RDBasis`: Represents a basis of quantum operators using Gell-Mann matrices.
- `RDCompositeBasis`: Represents a composite basis formed from multiple `RDBasis` objects.
- `RDSymbol`: Represents a custom symbol with an associated order for specific algebraic contexts.

Dependencies:
-------------
numpy==2.1.2
sympy==1.13.3
"""

from typing import Union

# Importing necessary modules from SymPy
from sympy.core.singleton import S
from sympy.physics.quantum import Operator, Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy import (
    Add, Pow, Expr, Matrix, Mul, Symbol, sqrt as sp_sqrt,
    kronecker_product, sympify, cancel, 
    MutableDenseMatrix, eye, I, sqrt, 
    Rational, nsimplify, zeros as sp_zeros,
    latex as sp_latex
)

from sympy.matrices.dense import matrix_multiply_elementwise as sp_elementwise

# Importing necessary modules from NumPy
from numpy import (
    array as np_array, append as np_append, 
    ndarray as np_ndarray, trace as np_trace, 
    vectorize as np_vectorize, any as np_any, 
    all as np_all, empty as np_empty, 
    concatenate as np_concatenate,
    isin as np_isin,
    sum as np_sum,
    zeros as np_zeros,
    logical_not as np_logical_not
)

# Importing additional utility from itertools
from itertools import product
        

class Blocks:
    """
    Handles groups of blocks (Block). Supports arithmetic operations.

    Attributes:
    -----------
    expr : ndarray
        A NumPy array containing the block expressions.
    
    Methods:
    --------
    simplify_blocks():
        Simplifies the blocks by aggregating those with similar deltas.
    
    hermitian():
        Returns the Hermitian conjugate of the blocks.
    
    __add__(self, other):
        Adds another `Block` or `Blocks` object to the current `Blocks` object.
    
    add_structure(structure):
        Adds a structure to each block in the `Blocks`.
    
    apply_mask(expr):
        Applies a mask to the given expression (either `MulGroup` or `Expression`) based on the blocks (self).
    """
    def __init__(self, expr=None):
        """
        Initializes the `Blocks` object.
        
        Parameters:
        ----------
        expr : np_ndarray, optional
            A NumPy array of expressions to initialize the block with. Default is an empty array.
        """
        self.expr = np_empty(0, dtype=object) if expr is None else expr
        self.expr = self.expr[self.expr != 0]

        if len(self.expr) > 1:
            self.expr = self.simplify_blocks()

    def simplify_blocks(self):
        """
        Simplifies the block collection by combining blocks that 
        have the same infinite parts (`inf`).

        Returns:
        --------
        Blocks
            A simplified version of the current block collection.
        """
        blocks = self.expr
        
        if not np_any(blocks):
            return Blocks()
        
        
        inf_part = np_vectorize(lambda x: x.inf)(blocks)                            # Extract the 'infinite' part (`inf`) of each block using vectorization. 
        fin_part = np_vectorize(lambda x: x.fin, otypes=[np_ndarray])(blocks)       # Extract the 'finite' part (`fin`) of each block, also using vectorization.
        deltas = np_vectorize(lambda x: x.deltas, otypes=[dict])(blocks)            # Extract the delta values (shifts/offsets in bosonic subspaces) for each block, stored in a dictionary.

        
        result_dict = {}
        for fin, inf, delta in zip(fin_part, inf_part, deltas):                     # Iterate through the `fin`, `inf`, and `delta` parts of each block.
            key = (inf, tuple(delta.items()))                                       # Create a unique key based on the `inf` (infinite part) and the `delta` (offsets).
            if key in result_dict:                                                  # If this key already exists in the result dictionary, add the `fin` (finite part) to the existing entry.
                result_dict[key] = result_dict[key] + fin
            else:                                                                   # Otherwise, create a new entry in the dictionary with the current `fin` part.
                result_dict[key] = fin


        # Convert the result dictionary back into a list of `Block` objects.
        # The key provides the `inf` and `delta`, and the value is the combined `fin`.
        return np_array([Block(fn, inf, dict(delta)) for (inf, delta), fn in result_dict.items()])
    
    def hermitian(self):
        """
        Returns the Hermitian of the current block collection.

        Returns:
        --------
        Blocks
            The Hermitian of the block collection.
        """
        return Blocks(np_array([block.hermitian() for block in self.expr]))
    
    def __add__(self, other):
        """
        Adds a `Block` or `Blocks` object to the current block collection.

        Parameters:
        ----------
        other : Block or Blocks
            The object to add.

        Returns:
        --------
        Blocks
            The resulting collection of blocks after addition.
        
        Raises:
        -------
        ValueError:
            If `other` is not a `Block` or `Blocks`.
        """
        if other == 0:
            return self
        if isinstance(other, Block):
            return Blocks(np_append(self.expr, other))
        if isinstance(other, Blocks):
            return Blocks(np_concatenate([self.expr, other.expr]))
        raise ValueError(f'Invalid type {type(other)} for addition. Must be Block or Blocks.')
    
    def add_structure(self, structure):
        """
        Applies a mask based on the given expression.

        Parameters:
        ----------
        expr : MulGroup or Expression
            The expression to mask.

        Returns:
        --------
        Expression
            A masked expression.
        """
        for block in self.expr:
            block.add_structure(structure)

    def apply_mask(self, expr:Union['MulGroup', 'Expression']):
        """
        Applies a mask based on 'self' to the given expression.

        Parameters:
        ----------
        expr : MulGroup or Expression
            The expression to which the mask will be applied. The mask determines which parts of 
            the expression should be included or excluded based on matching deltas.

        Returns:
        --------
        Expression
            The masked expression. It returns two `Expression` objects: 
                - One containing the parts of the expression that matched the mask.
                - One containing the parts that did not match.
        """
        blocks_deltas = np_array([x.new_deltas for x in self.expr])                         # Create an array of the `new_deltas` for all blocks in the current object (self).
                                                                                            # This is used for comparison with the delta values of the input expression.
        
        if isinstance(expr, MulGroup):                                                      # If the input expression is a single MulGroup (multiplicative group):
            truth_array = np_all(blocks_deltas == expr.delta, axis=1)                       # Create a boolean array (`truth_array`) where True indicates that the block's delta matches the MulGroup's delta.               
            total_trues = np_sum(truth_array)                                               # Sum the number of True values (i.e., the number of blocks that match the MulGroup's delta).

            if total_trues == 0:                                                            # If no blocks match the MulGroup's delta, return an empty `true_expr` and `false_expr` containing the original MulGroup.
                return Expression(), Expression(np_array([expr], dtype=object))

            if total_trues > 1:                                                             # If more than one block matches, raise an error because that means that something went wrong on the simplification rutine (Debugging purposes).
                raise ValueError(f'Multiple blocks: {self.expr[truth_array]}  have the same deltas as the MulGroup. If you saw this message, please report it to the developers.')
            
                                                                                            # Otherwise, take the first block that matches and apply the mask to the MulGroup.
            block : Block = self.expr[truth_array][0]
            return block.apply_mask(expr)

        if isinstance(expr, Expression):                                                    # If the input is an `Expression` (which contains multiple MulGroups):
            if len(expr.expr) == 0:                                                         # If the expression is empty (contains no MulGroups), return two empty expressions.
                return Expression(), Expression()
            if len(expr.expr) == 1:                                                         # If the expression has only one MulGroup, directly apply the mask to that single MulGroup.
                return self.apply_mask(expr.expr[0])
            
                                                                                            # Otherwise, extract the delta values of all MulGroups in the expression.
            mul_groups_deltas = np_array([x.delta for x in expr.expr])                      # Create a matrix where each row corresponds to whether a MulGroup's delta matches a block's delta.
            truth_matrix = np_all(mul_groups_deltas[:, None, :] == blocks_deltas, axis=2)   # This matrix will have True in cells where the delta of a block matches that of a MulGroup.
            
                                                                                            # Initialize two empty `Expression` objects to store the matching and non-matching parts of the input.
            true_expr = Expression()
            false_expr = Expression()

            for n_mul_group, truth_array in enumerate(truth_matrix):                        # Iterate over the truth matrix, checking for matching blocks for each MulGroup in the expression.
                mul_group = expr.expr[n_mul_group]                                          # Get the current MulGroup from the expression
                total_trues = np_sum(truth_array)                                           # Count how many blocks match this MulGroup's delta

                if total_trues == 0:                                                        # If no blocks match, add the MulGroup to the `false_expr` (i.e., non-matching parts).
                    false_expr += mul_group
                    continue

                elif total_trues > 1:                                                       # If more than one block matches, raise an error because this is ambiguous.
                    raise ValueError(f'Multiple blocks: {self.expr[truth_array]}  have the same deltas as the MulGroup. If you saw this message, please report it to the developers.')
                
                                                                                            # Otherwise, take the first matching block and apply the mask to this MulGroup.
                block : Block = self.expr[truth_array][0]
                true_Expr, false_Expr = block.apply_mask(mul_group)

                                                                                            # Add the results of the mask application to the appropriate expression.
                true_expr += true_Expr
                false_expr += false_Expr
        
            return true_expr, false_expr
        
    def _repr_latex_(self):
        
        if len(self.expr) == 0:
            return '$0$'
        return '$' + ' + '.join([expr._latex_() for expr in self.expr]) + '$'

        
            
class Block:
    """
    Represents an individual block with methods for manipulating and applying masks to the block.

    Attributes:
    -----------
    fin : ndarray
        The finite part of the block, represented as a boolean matrix.
    
    inf : Expr
        The infinite part of the block, represented as a symbolic expression.
    
    deltas : dict
        A dictionary representing the deltas.
    
    Methods:
    --------
    hermitian():
        Returns the Hermitian conjugate of the block.
    
    add_structure(structure):
        Adds a structure to the block for alignment with other blocks.
    
    apply_mask(expr):
        Applies a mask to the given expression based on the block (self).
    """
    
    def __init__(self, fin : Union[MutableDenseMatrix, np_ndarray]=None, inf: Expr=None, deltas=None):
        """
        Initialize a block with a finite and infinite part, as well as deltas.

        Parameters
        ----------
        fin : MutableDenseMatrix or np_ndarray
            The finite part of the block.
        
        inf : Expr, optional
            The infinite part of the block (default is None).
        
        deltas : dict, optional
            A dictionary representing deltas (default is None).
        """

        if fin is None and inf is None:
            raise ValueError('fin and inf cannot be both None')

        if fin is not None and not isinstance(fin, MutableDenseMatrix) and not isinstance(fin, np_ndarray):
            raise ValueError('fin must be a MutableDenseMatrix or a NumPy array')

        if fin is None:
            fin = [[1]]
    
        if not np_all(np_isin(fin, [0, 1, True, False])):
            raise ValueError('fin must be a MutableDenseMatrix (NumPy array) of only 0 (False) or 1 (True)')
        
        if inf is not None:
            inf = inf.expand()
            if isinstance(inf, Add):
                raise ValueError('inf must be a Mul')
            
            if not np_all(np_vectorize(lambda x: x.expand().has(BosonOp))(inf.as_ordered_factors())):
                raise ValueError('inf must be a Mul of only BosonOp')
            
        self.fin = np_array(fin, dtype=bool)
        self.fin_not = np_logical_not(self.fin)
        self.inf = inf

        if deltas is None and inf is not None:                                                      # If `deltas` is not provided and `inf` is given, compute the deltas based on the infinite part.
            self.deltas = {}
            for i, n in enumerate(inf.as_ordered_factors()):                                        # Iterate over the factors in the infinite part (`inf`), which are bosonic operators.
                a_op : BosonOp = n                                                                  # Assume the factor is a bosonic operator.
                a_exp = 1                                                                           # Initialize the exponent to 1.
                if isinstance(n, Pow):                                                              # If the factor is a power, extract the base and exponent.
                    a_op = n.base                                                                   # Update the operator.
                    a_exp = n.exp                                                                   # Update the exponent.                      
                n_op = Dagger(a_op) * a_op if a_op.is_annihilation else a_op * Dagger(a_op)         # Compute the number operator based on the bosonic operator.
                self.deltas[n_op] = self.deltas.get(n_op, 0) + (-1)**a_op.is_annihilation * a_exp   # Update the delta value based on the number operator and exponent.
        else:
            self.deltas = deltas if inf is not None else {}                                         # If `deltas` is provided, use it. Otherwise, initialize an empty dictionary.

        if self.has_diagonal():                                                                     # Check if the block contains a fully diagonal component.
            raise ValueError('This block contain a fully diagonal component. It should be a Block with only off-diagonal elements.')
        
    def hermitian(self):
        """
        Returns the Hermitian conjugate of the block.

        Returns
        -------
        Block
            A new `Block` object representing the Hermitian conjugate.
        """
        if self.inf is None:
            return Block(self.fin.T)
        return Block(self.fin.T, Dagger(self.inf))
        
    def has_diagonal(self):
        """
        Checks whether the block contains a fully diagonal component.

        Returns
        -------
        bool
            True if the block contains a diagonal component, otherwise False.
        """
        inf_diagonal = np_sum(list(self.deltas.values())) == 0
        fin_diagonal = np_sum(self.fin.diagonal()) != 0

        return inf_diagonal and fin_diagonal
    
    def add_structure(self, structure):
        """
        Adds a structure to the block to align it with other blocks.

        Parameters
        ----------
        structure : dict
            A dictionary mapping operators to structural positions within the delta array.
            {Dagger[BosonOp] * BosoOp: Position}
        """
        self.new_deltas = np_zeros(len(structure)) if structure != {} else np_array([0])                    # Initialize the new deltas based on the structure.

        for n_op, delta in self.deltas.items():                                                             # Iterate over the current deltas.
            self.new_deltas[structure[n_op]] = delta                                                        # Update the new deltas based on the structure.

        self.fin = Matrix(self.fin * 1)                                                                     # Update the finite part of the block.
        self.fin_not = Matrix(self.fin_not * 1)                                                             # Update the negation of the finite part.
    
    def apply_mask(self, expr:'MulGroup'):
        """
        Applies a mask to the given expression based on the block (self).

        Parameters
        ----------
        expr : MulGroup or Expression
            The expression to apply the mask to.

        Returns
        -------
        Expression
            A new `Expression` object after applying the mask.
        """
        # Assuming that this method is only called when the infinite part of the block matches the infinite part of the MulGroup.
        if isinstance(expr, MulGroup):
            fin = expr.fn
            fin_true = sp_elementwise(fin, self.fin)                                                  # Apply the mask to the finite part of the expression.
            fin_false = sp_elementwise(fin, self.fin_not)                                             # Apply the mask to the negation of the finite part.
            return Expression(np_array([MulGroup(fin_true, expr.inf, expr.delta, expr.Ns)], dtype=object)), Expression(np_array([MulGroup(fin_false, expr.inf, expr.delta, expr.Ns)], dtype=object))
    
    def __add__(self, other):
        """
        Adds another block to the current block.

        Parameters
        ----------
        other : Block or Blocks
            The object to add.

        Returns
        -------
        Block or Blocks
            The result of the addition.
        """
        if other == 0:
            return self
        if isinstance(other, Block):
            if other.inf == self.inf:
                return Block(self.fin + other.fin, self.inf, self.deltas)
            return Blocks(np_array([self, other]))
        if isinstance(other, Blocks):
            return other + self
        raise ValueError(f'Invalid type {type(other)} for addition. Must be Block or Blocks.')  
    
    def __radd__(self, other):
        return self.__add__(other)

    def _latex_(self):
        if self.inf is None:
            return f'{sp_latex(MutableDenseMatrix(self.fin*1))}'
        return f'{sp_latex(MutableDenseMatrix(self.fin*1))}  \\cdot {sp_latex(self.inf)}'
  
    def _repr_latex_(self):
        return f'${self._latex_()}$'
      


class Expression:
    """
    Represents a symbolic or numeric expression using NumPy arrays.
    
    Attributes:
    -----------
    expr : ndarray
        The internal representation of the expression as a NumPy array.
    
    Methods:
    --------
    __add__(self, other):
        Defines addition with another `Expression` or `MulGroup`.
        
    __sub__(self, other):full_diagonalization
        Defines subtraction.
        
    __neg__(self):
        Returns the negation of the current expression.
        
    __mul__(self, other):
        Defines multiplication with another `Expression` or a scalar.
        
    __rmul__(self, other):
        Defines reverse multiplication for cases like scalar * `Expression`.
        
    _repr_latex_(self):
        Generates a LaTeX string representation of the expression.
    """

    def __init__(self, expr=None):
        """
        Parameters
        ----------
        expr : ndarray, optional
            Initial expression represented as a NumPy array. Default is an empty array.
        """
        self.expr = np_empty(0, dtype=object) if expr is None else expr
        self.expr = self.expr[self.expr != 0]
    
    def __add__(self, other):
        """
        Adds another `Expression` or `MulGroup` to this one.

        Parameters
        ----------
        other : Expression or MulGroup
            The object to add.

        Returns
        -------
        Expression
            A new `Expression` representing the sum.

        Notes:
        ------
        If `other` is not an instance of `Expression` or `MulGroup`, a ValueError 
        will be raised indicating an invalid type for addition.
        """
        if isinstance(other, Expression):
            return Expression(np_concatenate([self.expr, other.expr]))
        if isinstance(other, MulGroup):
            return Expression(np_append(self.expr, other))
        if not np_any(other):
            return self
        raise ValueError('Invalid type for addition.')
    
    def simplify(self, return_dict=False):
        """
        Simplifies multiplicative groups in the given expression.

        Parameters
        ----------
        expr : Expression
            The expression containing multiplicative groups.
        return_dict : bool, optional
            If True, returns a dictionary of simplified groups instead of an Expression object.

        Returns
        -------
        Expression or dict
            A simplified Expression object or a dictionary of results based on the return_dict flag.
        """
        mul_groups = self.expr                                                                    # Get the multiplicative groups from the expression.
        if not np_any(mul_groups):                                                                # If there are no multiplicative groups, return an empty expression.
            return Expression()
        
        inf_part = np_vectorize(lambda x: x.inf, otypes=[object])(mul_groups)                 # Extract the infinite parts of each multiplicative group.
        fn_part = np_vectorize(lambda x: x.fn)(mul_groups)                                    # Extract the function parts of each multiplicative group.
        deltas = np_vectorize(lambda x: x.delta, otypes=[np_ndarray])(mul_groups)             # Extract the delta values of each multiplicative group.

        Ns = mul_groups[0].Ns                                                               # Get the number operators from the first multiplicative group.

        result_dict = {}
        for fn, inf, delta in zip(fn_part, inf_part, deltas):                               # Iterate over the function, infinite, and delta parts of each multiplicative group.
            key = (tuple(inf), tuple(delta))                                                # Create a unique key based on the infinite and delta parts.
            if key in result_dict:                                                          # If the key already exists in the result dictionary, add the function to the existing entry.
                result_dict[key] = result_dict[key] + fn                                    # Update the dictionary with the new function.
            else:
                result_dict[key] = fn                                                       # Otherwise, create a new entry with the current function.
        
        if return_dict:                                                                     # If the return_dict flag is set, return the dictionary of simplified groups.
            return result_dict                                                              # Return the dictionary of simplified groups.
                
        return Expression(np_array([MulGroup(fn, np_array(inf), np_array(delta), Ns) for (inf, delta), fn in result_dict.items()]))

    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return Expression(-self.expr)
    
    def __mul__(self, other):
        """
        Multiplies this expression with another `Expression` or a term

        Parameters
        ----------
        other : Expression, or term
            The object to multiply with.

        Returns
        -------
        Expression
            A new `Expression` representing the product.

        Notes:
        ------
        If `other` is not an instance of `Expression`, it is assumed to be a term 
        value for multiplication.
        """
        if isinstance(other, Expression):
            result_expression = (self.expr[None, :] * other.expr[:, None]).flatten()
            return Expression(result_expression)
        return Expression(self.expr * other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def _repr_latex_(self):
        if len(self.expr) == 0:
            return '$0$'
        return '$' + ' + '.join([sp_latex(expr) for expr in self.expr]) + '$'
    

class MulGroup(Expr):
    """
    Represents a multiplicative group used for symbolic matrix multiplication.
    
    Attributes:
    -----------
    fn : Expr
        The matrix of functions of number operators associated with the group.
    
    inf : ndarray[Mul[BosonOp]]
        The multiplicative bosonic operator for the group.
    
    delta : int
        Represents a shift or offset for the group.
    
    Ns : ndarray
        A NumPy array representing the elements of the group.
    
    Methods:
    --------
    __neg__(self):
        Returns the negation of the group.
        
    __add__(self, other):
        Adds another `MulGroup` or `Expression` to the current group.
        
    __sub__(self, other):
        Subtracts another `MulGroup` or `Expression`.
        
    __mul__(self, other):
        Defines multiplication with another `MulGroup` or `Expression`.
        
    __rmul__(self, other):
        Defines reverse multiplication for the group.
        
    _sympystr(self, printer):
        Returns a string representation of the group.
        
    _latex(self, printer):
        Returns a LaTeX representation of the group.
    """

    
    @property
    def fn(self):
        """
        Returns the matrix of functions of number operators associated with the group.

        Returns:
        --------
        Matrix:
            The matrix of functions of number operators associated with the group.
        """
        return self.args[0]
    
    @property
    def inf(self):
        """
        Returns the multiplicative bosonic operator for the group.

        Returns:
        --------
        ndarray[Mul[BosonOp]]:
            The multiplicative bosonic operator for the group.
        """
        return self.args[1]
    
    @property
    def delta(self):
        """
        Returns the shift or offset from the diagonal of each bosoic subspace.

        Returns:
        --------
        ndarray[int]:
            The delta values representing the shift or offset from the diagonal.
        """
        return self.args[2]
    
    @property
    def Ns(self):
        """
        Returns the array representing the number operators for each bosonic subspace.

        Returns:
        --------
        ndarray
            The elements of the group representing the number operators.
        """
        return self.args[3]
    
    def __new__(cls, fn, inf=[1], delta=[0], Ns=np_ndarray):
        """
        Parameters
        ----------
        fn : Expr
            The matrix of functions of number operators associated with the group.
        inf : int or float, optional
            A multiplicative scalar factor for the group. Default is 1.
        delta : int, optional
            Represents a shift or offset for the group. Default is 0.
        Ns : ndarray, optional
            A NumPy array representing the number operators for each bosonic subspace. Default is np_ndarray.

        Returns
        -------
        MulGroup
            A new `MulGroup` object.
        
        Notes:
        ------
        The `__new__` method checks for the validity of the provided parameters and 
        creates an instance of `MulGroup`, returning a zero expression if any of the 
        initial parameters are invalid (e.g., `fn` or `inf` being empty).
        """
        if not np_any(fn) or not np_any(inf):
            return S.Zero
        if not isinstance(fn, MutableDenseMatrix):
            fn = MutableDenseMatrix([fn])
        obj = Expr.__new__(cls, fn, inf, delta, Ns)

        return obj

    def __neg__(self):
        return MulGroup(-self.fn, self.inf, self.delta, self.Ns)
    
    def __add__(self, other):
        """
        Adds another `MulGroup` or `Expression` to the current group.
        
        Parameters
        ----------
        other : MulGroup or Expression
            The object to add.

        Returns
        -------
        MulGroup or Expression
            A new `MulGroup` or `Expression` representing the sum.

        Notes:
        ------
        If `other` is an instance of `Expression`, it will be converted to an 
        `Expression` and added to the current group.
        """
        if isinstance(other, MulGroup) and np_all(other.inf == self.inf):
            return MulGroup(self.fn + other.fn, self.inf, self.delta, self.Ns)
        if isinstance(other, Expression):
            # Other is a numpy array of MulGroups -> Expression
            return other + self
        return Expression(np_array([self, other], dtype=object))
    
    def __sub__(self, other):
        return (self + (-other))

    def __mul__(self, other):
        """
        Multiplies this group with another `MulGroup` or `Expression`.

        Parameters
        ----------
        other : MulGroup, Expression, or scalar
            The object to multiply with.

        Returns
        -------
        MulGroup or Expression
            A new `MulGroup` or `Expression` representing the product.

        Notes:
        ------
        If `other` is not a `MulGroup`, it is treated as a scalar and multiplied accordingly.
        """
        if isinstance(other, MulGroup):
            mul_inf = self.inf * other.inf                                      # Multiply the infinite parts of the groups.
            xreplace_dict = dict(zip(self.Ns, self.Ns - self.delta))            # Create a dictionary for replacing the number operators with the shifted values.
            mul_fn = self.fn * other.fn.xreplace(xreplace_dict)
            return MulGroup(mul_fn, mul_inf, self.delta + other.delta, self.Ns)
        if isinstance(other, Expression):
            # Other is a numpy array of MulGroups -> Expression
            return Expression(self * other.expr)
        return MulGroup(self.fn * other, self.inf, self.delta, self.Ns)

    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def _sympystr(self, printer):
        return f'{self.fn} * {Mul(*self.inf)}'
    def _latex(self, printer):
        return f'{printer._print(self.fn)}  \\cdot {printer._print(Mul(*self.inf))}'

        

class RDOperator(Operator):
    """
    Represents a quantum operator with a matrix and associated subspace.
    
    Attributes:
    -----------
    name : str
        The name of the operator.
    
    matrix : Matrix
        The matrix representation of the operator.
    
    subspace : str
        The subspace in which the operator acts.
    
    Methods:
    --------
    _sympystr(self, printer):
        Returns a string representation of the operator.
        
    _latex(self, printer):
        Returns a LaTeX representation of the operator.
    """


    @property
    def name(self):
        return self.args[0]
    
    @property
    def matrix(self):
        M = Matrix(self.args[1])
        dim = int(sp_sqrt(M.shape[0]))
        M = M.reshape(dim, dim)
        self.is_identity = M == eye(dim)
        return M
    
    @property
    def subspace(self):
        return self._subspace

    def __new__(cls, name, matrix, subspace='default'):
        """
        Creates a new instance of the `RDOperator` class.

        Parameters
        ----------
        name : str
            The name of the operator.
        matrix : Matrix
            The matrix representation of the operator.
        subspace : str, optional
            The subspace in which the operator acts. Default is 'default'.

        Returns
        -------
        RDOperator
            A new `RDOperator` instance.
        """
        matrix = sympify(matrix)
        obj = Operator.__new__(cls, name, matrix)
        # obj._matrix = matrix
        obj._subspace = subspace
        return obj
        
    def _sympystr(self, printer):
        return printer._print(self.name)
    
    def _latex(self, printer):
        return printer._print(self.name)

class RDBasis:
    """
    Represents a basis of quantum operators using Gell-Mann matrices.
    
    Attributes:
    -----------
    name : str
        The name of the basis.
    
    dim : int
        The dimension of the basis (number of elements).
    
    basis : ndarray
        The array of `RDOperator` objects representing the basis.
    
    basis_matrices : ndarray
        The array of matrix representations of the basis elements.
    
    basis_ling_alg_norm : int or float
        A normalization factor for the algebraic properties of the basis.
    
    Methods:
    --------
    get_gell_mann(self):
        Generates Gell-Mann matrices to form the basis.
        
    project(self, to_be_projected):
        Projects a given matrix onto the basis.
        
    Raises:
    -------
    ValueError:
        If the matrix to be projected has incorrect dimensions.
    """

    def __init__(self, name:str, dim:int):
        """
        Parameters
        ----------
        name : str
            The name of the basis.
        dim : int
            The dimension of the basis.
        """
        self.name = name
        self.dim = dim
        matrix_basis = self.get_gell_mann()
        self.basis = np_array([RDOperator(f'{name}_{i}', mat, subspace=name)
                             for i, mat in enumerate(matrix_basis)], dtype=object)
        self.basis_matrices = np_array([basis.matrix for basis in self.basis], dtype=object)
        if len(self.basis) == 1:
            self.basis_ling_alg_norm = dim
        else:
            self.basis_ling_alg_norm = nsimplify((self.basis[1].matrix.T.conjugate() @ self.basis[1].matrix).trace())
            
    
    def get_gell_mann(self):
        """
        Generates the Gell-Mann matrices used in the basis.

        Returns
        -------
        list[Matrix]
            A list of Gell-Mann matrices forming the basis.
        """
        matrices = [eye(self.dim)]

        # Lambda_1 to Lambda_(n-1)^2
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                # Symmetric Gell-Mann matrices
                symm = sp_zeros(self.dim, self.dim)
                symm[i, j] = 1
                symm[j, i] = 1
                matrices.append(symm)

                # Anti-symmetric Gell-Mann matrices
                asymm = sp_zeros(self.dim, self.dim)
                asymm[i, j] = -I
                asymm[j, i] = I
                matrices.append(asymm)

        # Diagonal Gell-Mann matrices
        for k in range(1, self.dim):
            diag = sp_zeros(self.dim, self.dim)
            for i in range(k):
                diag[i, i] = 1
            diag[k, k] = -k
            diag = diag * sqrt(Rational(2, (k * (k + 1))))
            matrices.append(diag)

        return matrices

    def project(self, to_be_projected: Matrix):
        """
        Projects a given matrix onto the basis.

        Parameters
        ----------
        to_be_projected : Matrix
            The matrix to be projected onto the basis.

        Returns
        -------
        Expression
            A symbolic expression representing the projection.
        
        Raisesis_identity
        ------
        ValueError
            If the matrix to be projected has incorrect dimensions.
        """
        if to_be_projected.shape != (self.dim, self.dim):
            raise ValueError('Matrix to be projected has wrong shape.')

        basis_coeffs = np_vectorize(cancel)(np_trace([matrix.T.conjugate() @ to_be_projected for matrix in self.basis_matrices], axis1=1, axis2=2))     # Compute the coefficients of the input matrix in terms of the basis elements.
        basis_coeffs /= self.basis_ling_alg_norm                                                                                                  # Normalize the coefficients using the `basis_ling_alg_norm`.
        basis_coeffs[0] *= self.basis_ling_alg_norm / self.dim                                                                                   # Update the first coefficient based on the normalization factor.
        if basis_coeffs[0] == 1 and all(basis_coeffs[1:] == 0):                                                                                 # If the first coefficient equals 1 and all other coefficients are zero, return 1.
            return 1
        return basis_coeffs.dot(self.basis)
    

class RDCompositeBasis:
    """
    Represents a composite basis formed from multiple `RDBasis` objects.

    Attributes:
    -----------
    bases : list of RDBasis
        A list of `RDBasis` objects forming the composite basis.
    
    dim : int
        The total dimension of the composite basis.
    
    basis : ndarray
        The array of symbolic composite basis elements.
    
    basis_matrices : ndarray
        The array of matrix representations for the composite basis elements.
    
    basis_ling_alg_norm : int or float
        A normalization factor for the composite basis.
    
    Methods:
    --------
    project(self, to_be_projected):
        Projects a given matrix onto the composite basis.

    __init__(self, bases: list[RDBasis]):
        Initializes the RDCompositeBasis with a list of RDBasis objects.

    Parameters:
    -----------
    bases : list[RDBasis]
        A list of RDBasis objects to form the composite basis. The total dimension 
        is calculated as the product of the dimensions of each `RDBasis` object 
        in the list.

    Notes:
    ------
    The composite basis is formed by taking the Kronecker product of the 
    individual basis matrices and creating symbolic basis elements as 
    products of the basis elements of the constituent `RDBasis` objects.
    
    The normalization factor for the composite basis is computed based on 
    the trace of the matrix representation of the basis elements.
    """

    def __init__(self, bases: list[RDBasis]):
        """
        Initializes the RDCompositeBasis with a list of RDBasis objects.

        Parameters:
        -----------
        bases : list[RDBasis]
            A list of `RDBasis` objects to form the composite basis. The total 
            dimension is calculated as the product of the dimensions of each 
            `RDBasis` object in the list.
        """
        self.bases = bases
        self.dim = Mul(*[basis.dim for basis in bases])
        self.basis = []
        
        self.basis_matrices = []
        for p in product(*[basis.basis for basis in bases]):                                    # Iterate over the Cartesian product of the basis elements.
            self.basis.append(Mul(*[1 if p_.is_identity else p_ for p_ in p]))                  # Create the symbolic composite basis elements.
            self.basis_matrices.append(kronecker_product(*[basis.matrix for basis in p]))       # Create the matrix representations of the composite basis elements.

        self.basis = np_array(self.basis)                                                    # Convert the basis to a NumPy array.
        self.basis_matrices = np_array(self.basis_matrices, dtype=object)                 # Convert the basis matrices to a NumPy array.

        if len(self.basis) == 1:
            self.basis_ling_alg_norm = self.dim
        else:
            self.basis_ling_alg_norm = nsimplify((self.basis_matrices[1].T.conjugate() @ self.basis_matrices[1]).trace())   # Compute the normalization factor based on the trace of the matrix representation of the basis elements.
            
        
    def project(self, to_be_projected):
        """
        Projects a given matrix onto the composite basis.

        Parameters:
        -----------
        to_be_projected : ndarray
            The matrix to be projected onto the composite basis. It should be compatible 
            with the dimensions of the composite basis.

        Returns:
        --------
        ndarray or int:
            The projected representation of the input matrix in the composite basis. 
            If the first coefficient equals 1 and all other coefficients are zero, 
            returns 1; otherwise, returns the expanded result of the basis coefficients 
            dot product with the composite basis.

        Notes:
        ------
        The projection is performed by computing the coefficients of the input matrix 
        in terms of the composite basis elements, normalizing the coefficients using 
        the `basis_ling_alg_norm`.
        """
        basis_coeffs = np_vectorize(cancel)(np_trace([matrix.T.conjugate() @ to_be_projected for matrix in self.basis_matrices], axis1=1, axis2=2))    # Compute the coefficients of the input matrix in terms of the composite basis elements.

        basis_coeffs /= self.basis_ling_alg_norm                                                                                                 # Normalize the coefficients using the `basis_ling_alg_norm`.
        basis_coeffs[0] *= self.basis_ling_alg_norm / self.dim                                                                                # Update the first coefficient based on the normalization factor.

        if basis_coeffs[0] == 1 and all(basis_coeffs[1:] == 0):                                                                             # If the first coefficient equals 1 and all other coefficients are zero,
            return 1

        return basis_coeffs.dot(self.basis).expand()
    


class RDSymbol(Symbol):
    """
    Represents a custom symbol with an associated order.
    
    Attributes:
    -----------
    order : int
        The order of the symbol (used in specific algebraic contexts).
    
    Methods:
    --------
    __new__(cls, name, *args, order=0, **kwargs):
        Creates a new instance of the `RDSymbol` class.
    """

    @property
    def order(self):
        return self._order

    def __new__(cls, name, *args, order=0, **kwargs):
        obj = Symbol.__new__(cls, name, *args, **kwargs)
        if isinstance(order, complex):
            raise ValueError('Order must be real.')
        obj._order = order
        return obj
