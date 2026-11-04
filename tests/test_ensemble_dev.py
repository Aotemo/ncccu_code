"""Independent ranking examples for the fixed two-member blend decision."""
import importlib.util
import os
from pathlib import Path
import sys


def function(name='logit_grid'):
    source=Path(os.environ.get('NCCCU_TEST_SOURCE',Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(source/'runtime/bundle'))
    path=Path(__file__).resolve().parents[1]/'tools/diagnostics/ensemble_dev.py'
    assert path.is_file(), 'F01 fixed logit grid behavior is missing'
    spec=importlib.util.spec_from_file_location('ensemble_dev',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert callable(getattr(module,name,None)), 'requested fusion decision behavior is missing: '+name
    return getattr(module,name)


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


def test_equal_standardization_removes_member_scale_and_repairs_complementary_errors():
    result=function('standardized_equal_schemes')([1,1,0,0],
        {'left':[-1,3,0,0],'right':[30,-10,0,0]},['left','right'],top_k=2)
    assert len(result['schemes'])==1
    assert result['normalization']['left']=={'mean':.5,'std':1.5}
    assert result['normalization']['right']=={'mean':5.,'std':15.}
    assert result['selected']['auroc']==result['selected']['ap']==1
    assert result['selected']['weights']=={'left':.5,'right':.5}


def test_fixed_top_k_uses_single_member_ranking_and_registered_tie_order():
    result=function('standardized_equal_schemes')([1,0],
        {'A':[4,1],'B':[2,0],'bad':[0,3]},['B','A','bad'],top_k=2)
    top=next(row for row in result['schemes']if row['id']=='top_k_equal_z')
    assert top['members']==['B','A']and top['weights']=={'B':.5,'A':.5}
    assert result['selected']['id']=='top_k_equal_z'


def test_zero_variance_cannot_silently_enter_standardized_fusion():
    try:
        function('standardized_equal_schemes')([1,0],{'a':[1,0],'constant':[2,2]},['a','constant'])
    except ValueError as error:
        assert 'variance'in str(error)
    else:
        raise AssertionError('zero variance member was accepted')


def test_historical_load_receipt_requires_the_same_selected_identity_and_sidecar():
    import hashlib,json,tempfile
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);blob=root/'checkpoint.pt';blob.write_bytes(b'checkpoint')
        side=root/'checkpoint.pt.json';side.write_text('{"sha256":"blob"}')
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        checkpoint={'path':str(blob),'sidecar':{'path':str(side),'sha256':digest(side)}}
        freeze={'path':'freeze','sha256':'frozen'}
        row={'spec':{'freeze':freeze},'selected_blob_sha256':'blob','selected_tensor_sha256':'tensor',
             'selected':{'variant':'ema','successful_updates':3200},'selected_checkpoint_candidates':[checkpoint]}
        receipt={'status':'actual_blob_and_selected_tensor_loaded_with_original_frozen_runtime','member':'R154',
                 'freeze':freeze,'blob_sha256':'blob','selected_tensor_sha256':'tensor','variant':'ema',
                 'successful_updates':3200,'checkpoint':checkpoint,'frozen_source_files_verified':59,
                 'model_constructed':False,'forward_calls':0,'GPU_initialized':False,'official_read':False,
                 'source_commit':'source'}
        path=root/'receipt.json';path.write_text(json.dumps(receipt));load=function('verify_historical_load_receipt')
        assert load('R154',row,path,digest(path))['tensor_sha256']=='tensor'
        receipt['selected_tensor_sha256']='wrong';path.write_text(json.dumps(receipt))
        try:load('R154',row,path,digest(path))
        except ValueError as error:assert 'identity differs'in str(error)
        else:raise AssertionError('mismatched historical tensor accepted')


if __name__=='__main__':
    test_complementary_errors_improve_and_tie_prefers_incumbent_weight()
    test_no_gain_keeps_incumbent_vertex()
    test_equal_standardization_removes_member_scale_and_repairs_complementary_errors()
    test_fixed_top_k_uses_single_member_ranking_and_registered_tie_order()
    test_zero_variance_cannot_silently_enter_standardized_fusion()
    test_historical_load_receipt_requires_the_same_selected_identity_and_sidecar()
    print('six independent F01/F02 ranking and receipt targets passed',flush=True)
