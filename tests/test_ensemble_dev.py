"""Independent ranking examples for the fixed two-member blend decision."""
import importlib.util
import os
from pathlib import Path
import sys


def function():
    source=Path(os.environ.get('NCCCU_TEST_SOURCE',Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(source/'runtime/bundle'))
    path=Path(__file__).resolve().parents[1]/'tools/diagnostics/ensemble_dev.py'
    assert path.is_file(), 'F01 fixed logit grid behavior is missing'
    spec=importlib.util.spec_from_file_location('ensemble_dev',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert callable(getattr(module,'logit_grid',None)), 'F01 fixed logit grid behavior is missing'
    return module.logit_grid


def test_complementary_errors_improve_and_tie_prefers_incumbent_weight():
    result=function()([1,1,0,0],[-1,3,0,0],[3,-1,0,0])
    assert len(result['grid'])==21
    assert result['grid'][0]['auroc']==result['grid'][-1]['auroc']==.5
    assert result['selected']['auroc']==result['selected']['ap']==1
    assert result['selected']['R238_weight']==.3


def test_no_gain_keeps_incumbent_vertex():
    result=function()([1,1,0,0],[2,3,0,-1],[2,3,0,-1])
    assert result['selected']['R238_weight']==0
    assert result['selected']['auroc']==result['selected']['ap']==1


if __name__=='__main__':
    test_complementary_errors_improve_and_tie_prefers_incumbent_weight()
    test_no_gain_keeps_incumbent_vertex()
    print('two independent F01 ranking decisions passed',flush=True)
