import numpy as np
import pandas as pd
import pytest
from pet_scout.data import normalize_sequence
from pet_scout.features import alignment_identity, sequence_groups, composition
from pet_scout.evaluate import grouped_folds, validate_split, make_folds, selection_metrics, select_indices, fit_ridge


def test_sequence_validation():
    assert normalize_sequence(' acd\nef ')=='ACDEF'
    for sequence in ['', 'AXC', 'ACD*', None]:
        with pytest.raises(ValueError): normalize_sequence(sequence)


def test_identity_counts_gaps():
    assert alignment_identity('ACDEFGHIK','ACDEFGHIK')==1
    assert alignment_identity('ACDEFGHIK','ACDEFGHIKAAA')<1
    assert alignment_identity('ACDEFGHIK','ACDEYGHIK')<1


def test_transitive_groups():
    # A~B, B~C, but A not~C: all must stay together.
    identity=np.array([[1,.8,.4,0],[.8,1,.8,0],[.4,.8,1,0],[0,0,0,1]])
    groups=sequence_groups(identity,.6)
    assert groups[0]==groups[1]==groups[2]
    assert groups[3]!=groups[0]
    for train,test in grouped_folds(groups,2,11):
        validate_split(train,test,groups,identity,.6,True)


def test_group_split_reproducibility_and_partition():
    groups=np.repeat(np.arange(12),[1,2,3,4,2,3,1,2,4,1,3,2])
    a=grouped_folds(groups,5,11);b=grouped_folds(groups,5,11)
    assert sorted(np.concatenate([te for _,te in a]))==list(range(len(groups)))
    for (tr,te),(tr2,te2) in zip(a,b):
        assert np.array_equal(tr,tr2) and np.array_equal(te,te2)
        assert not set(groups[tr])&set(groups[te])


def test_identity_leak_rejected():
    with pytest.raises(ValueError,match='Cross-split'):
        validate_split(np.array([0]),np.array([1]),np.array([0,1]),np.ones((2,2)),.6,True)


def test_random_split_partition():
    folds=make_folds(20,np.arange(20),'random',5,3)
    assert sorted(np.concatenate([b for _,b in folds]))==list(range(20))


def test_selection_uses_prediction_not_outcome():
    y=np.array([100.,1.,2.]);p=np.array([0.,4.,2.]);ids=np.array(['a','b','c'])
    m=selection_metrics(y,p,ids,1)
    assert m['selected_ids']=='b' and m['selected_mean']==1
    assert m['oracle_mean']==100
    assert m['random_expected_mean']==pytest.approx(103/3)
    assert m['opportunity_captured']<0


def test_ties_are_outcome_independent():
    assert select_indices([2,2,2],['c','a','b'],2).tolist()==[1,2]
    assert selection_metrics(np.zeros(3),np.zeros(3),np.array(['a','b','c']),1)['opportunity_captured'] is None


def test_held_out_labels_do_not_change_fit():
    rng=np.random.default_rng(12);x=rng.normal(size=(24,8));y=rng.uniform(0,100,24)
    train=np.arange(18);test=np.arange(18,24);groups=np.arange(24)
    a,alpha=fit_ridge(x,y,train,test,groups,'grouped',11,[10.,100.])
    y[test]=100000
    b,beta=fit_ridge(x,y,train,test,groups,'grouped',11,[10.,100.])
    np.testing.assert_array_equal(a,b);assert alpha==beta


def test_composition_is_sequence_only():
    table=pd.DataFrame({'sequence':['ACDE','AAAA'],'activity':[1,999]})
    x=composition(table);table['activity']=0
    np.testing.assert_array_equal(x,composition(table))
    np.testing.assert_allclose(x[:,:20].sum(axis=1),1)


def test_matched_random_preserves_grouped_fold_sizes():
    groups=np.repeat(np.arange(8),[12,5,2,1,1,1,1,1]);n=len(groups)
    grouped=make_folds(n,groups,'grouped',5,11)
    matched=make_folds(n,groups,'matched',5,11)
    assert [(len(tr),len(te)) for tr,te in grouped]==[(len(tr),len(te)) for tr,te in matched]
    assert sorted(np.concatenate([te for _,te in matched]))==list(range(n))
