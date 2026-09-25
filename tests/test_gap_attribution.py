import numpy as np
import pandas as pd
import pytest

from reviewer_revision.run_gap_attribution import binary, decompose, gap, paired_values
from exposure_observability import phase_statistic


def direct_gap(p, q, y=0):
    r,s=np.sqrt(p),np.sqrt(q)
    c=float(r@s)
    tangent=s-c*r
    sine=np.linalg.norm(tangent)
    theta=np.arctan2(sine,c)
    direction=np.eye(len(p))[y]-r[y]*r
    direction/=np.linalg.norm(direction)
    R=2*theta*float(tangent@direction)/sine if sine else 0.
    B=2*(np.arcsin(s[y])-np.arcsin(r[y]))
    h=float(np.sqrt(p[1:]/(1-p[0]))@np.sqrt(q[1:]/(1-q[0])))
    return R-B,h,R


def test_gap_against_full_distribution_spherical_projection():
    rng=np.random.default_rng(90071)
    for _ in range(500):
        p,q=rng.dirichlet(np.ones(9),size=2)
        expected,h,_=direct_gap(p,q)
        assert gap(p[0],q[0],h)==pytest.approx(expected,abs=3e-13)
        assert gap(p[0],q[0],h)>=-3e-13


def test_canonical_unchanged_realized_probability_example():
    p=np.array([.2,.3,.5]);q=np.array([.2,.6,.2])
    expected,h,_=direct_gap(p,q)
    assert expected==pytest.approx(.06071740612887654,abs=1e-14)
    assert float(gap(.2,.2,h))==pytest.approx(expected,abs=1e-14)


def test_binary_submanifold_zero_gap_including_near_boundaries():
    a=np.array([1e-12,.2,.999999999999,.5])
    b=np.array([.999999999999,.8,1e-12,.5])
    np.testing.assert_array_equal(gap(a,b,np.ones(4)),np.zeros(4))


def test_replacement_order_decomposition_and_null_groups():
    rng=np.random.default_rng(13579)
    a0,b0,a1,b1=rng.uniform(.001,.999,size=(4,100))
    h0,h1=rng.uniform(0,1,size=(2,100))
    d=decompose(a0,b0,h0,a1,b1,h1)
    np.testing.assert_allclose(d['H']+d['P'],d['delta_F'],atol=1e-14)
    fixed_h=decompose(a0,b0,h0,a1,b1,h0)
    np.testing.assert_array_equal(fixed_h['H'],np.zeros(100))
    fixed_x=decompose(a0,b0,h0,a0,b0,h1)
    np.testing.assert_array_equal(fixed_x['P'],np.zeros(100))
    reverse=decompose(a1,b1,h1,a0,b0,h0)
    np.testing.assert_allclose(reverse['H'],-d['H'],atol=1e-14)
    np.testing.assert_allclose(reverse['P'],-d['P'],atol=1e-14)


def synthetic_cache(p,q,r_error=0.):
    _,_,R=direct_gap(p,q)
    c=float(np.sqrt(p)@np.sqrt(q));L=2*np.arccos(c)
    lp=np.full((2,24),np.log(q[0]));lp[:,0]=np.log(p[0])
    alr=np.zeros((2,23,3));alr[:,0,1]=L;alr[:,0,2]=R+r_error
    mask=np.zeros((2,23),bool);mask[:,0]=True
    return lp,alr,mask


def test_pairing_masks_and_explicit_rounding_residual_bridge():
    base=synthetic_cache(np.array([.2,.3,.5]),np.array([.3,.5,.2]),r_error=1e-7)
    final=synthetic_cache(np.array([.25,.35,.4]),np.array([.4,.1,.5]),r_error=3e-7)
    rows,_,_=paired_values(base,final)
    for row in rows:
        assert row['delta_e']==pytest.approx(2e-7,abs=1e-14)
        assert row['delta_G']==pytest.approx(row['H']+row['P']+row['delta_e'],abs=1e-14)
    bad=list(final);bad[2]=bad[2].copy();bad[2][0,0]=False;bad[2][0,1]=True
    with pytest.raises(ValueError,match='masks'):
        paired_values(base,tuple(bad))


def test_linear_aggregation_preserves_slope_bridge():
    rng=np.random.default_rng(17);rows=[]
    for target in range(3):
        for block in range(8):
            for k in [0,1,2,4,8,16]:
                H,P,e=rng.normal(size=3)
                rows.append(dict(target_id=str(target),block_id=block,d=np.log2(k+1),
                                 H=H,P=P,e=e,G=H+P+e))
    f=pd.DataFrame(rows)
    assert phase_statistic(f,'G')==pytest.approx(sum(phase_statistic(f,k) for k in ['H','P','e']),abs=1e-14)


def test_invalid_hybrid_affinity_fails_closed():
    with pytest.raises(ValueError):
        gap(.1,.2,1.00001)
