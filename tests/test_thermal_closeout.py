import numpy as np
from scripts.run_thermal_closeout import onset

def test_recovery_onset_does_not_include_confirmation_delay():
    values=np.array([33,32.2,31.9]+[31.8]*9)
    assert onset(values,32,10)==2

def test_recovery_requires_ten_samples_and_can_be_censored():
    assert onset(np.array([33]+[31.0]*9),32,10) is None

def test_later_violation_prevents_early_recovery():
    values=np.array([31.0]*5+[33.0]+[31.0]*10)
    assert onset(values,32,10)==6
