from sympt import *


def test_RDSymbol():
    x = RDSymbol('x', real=True, order=2)
    assert x.name == 'x'
    assert x.is_real
    assert x.order == 2

    z = RDSymbol('z', real=False, order=1)
    assert z.name == 'z'
    assert z.order == 1
    assert not z.is_real
    assert list(group_by_order(z.conjugate()).keys())[0] == 1

def test_RDBasis():
    Spin = RDBasis('sigma', 2)
    assert Spin.name == 'sigma'

    s0, s1, s2, s3 = Spin.basis
    assert str(s0.name) == 'sigma_0'
    assert str(s1.name) == 'sigma_1'
    assert str(s2.name) == 'sigma_2'
    assert str(s3.name) == 'sigma_3'

    assert s0.matrix == Matrix([[1, 0], [0, 1]])
    assert s1.matrix == Matrix([[0, 1], [1, 0]])
    assert s2.matrix == Matrix([[0, -I], [I, 0]])
    assert s3.matrix == Matrix([[1, 0], [0, -1]])

    assert (Spin.project(Matrix([[1, 0], [0, 0]])) - Rational(1, 2) * (s0 + s3)).cancel() == 0
    assert (Spin.project(Matrix([[0, 1], [0, 0]])) - Rational(1, 2) * (s1 + I * s2)).cancel() == 0
    assert (Spin.project(Matrix([[0, 0], [1, 0]])) - Rational(1, 2) * (s1 - I * s2)).cancel() == 0
    assert (Spin.project(Matrix([[0, 0], [0, 1]])) - Rational(1, 2) * (s0 - s3)).cancel() == 0

    Projectors = RDBasis('P', 2, projector_form=True)
    P0, P1, P2, P3 = Projectors.basis

    assert str(P0.name) == 'P_0'
    assert str(P1.name) == 'P_1'
    assert str(P2.name) == 'P_2'
    assert str(P3.name) == 'P_3'

    assert P0.matrix == Matrix([[1, 0], [0, 0]])
    assert P1.matrix == Matrix([[0, 1], [0, 0]])
    assert P2.matrix == Matrix([[0, 0], [1, 0]])
    assert P3.matrix == Matrix([[0, 0], [0, 1]])

    assert (Projectors.project(Matrix([[1, 0], [0, 0]])) - P0).cancel() == 0
    assert (Projectors.project(Matrix([[0, 1], [0, 0]])) - P1).cancel() == 0
    assert (Projectors.project(Matrix([[0, 0], [1, 0]])) - P2).cancel() == 0
    assert (Projectors.project(Matrix([[0, 0], [0, 1]])) - P3).cancel() == 0

def test_RDCompositeBasis():
    Spin = RDBasis('sigma', 2)
    Projectors = RDBasis('P', 2, projector_form=True)
    Composite = RDCompositeBasis([Spin, Projectors])

    expr = kronecker_product(Spin.basis[0].matrix, Projectors.basis[0].matrix)
    assert (Composite.project(expr) - Projectors.basis[0]).cancel() == 0

def test_MulGroup():
    omega = RDSymbol('omega', real=True, order=0)
    a = BosonOp('a')
    ad = Dagger(a)

    omega_z = RDSymbol('omega_z', real=True, order=1)
    b = BosonOp('b')
    bd = Dagger(b)

    x = MulGroup(fn=omega   * Matrix([[1, 0], [0, 1]]) , inf= np_array([ad*a, 1]), delta = np_array([0, 0]), Ns=np_array([ad*a, bd*b]))
    y = MulGroup(fn=omega_z * Matrix([[1, 0], [0, -1]]), inf= np_array([1, bd*b]), delta = np_array([0, 0]), Ns=np_array([ad*a, bd*b]))

    multiplication = x * y
    assert (multiplication.fn -  omega * omega_z * Matrix([[1, 0], [0, -1]])) == sp_zeros(2)
    assert np_all(multiplication.inf == [ad*a, bd*b])
    assert np_all(multiplication.delta == [0, 0])
    assert np_all(multiplication.Ns == [ad*a, bd*b])
    