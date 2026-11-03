"""Register the already verified author native asset; no download/conversion/forward."""
import argparse
import json
import sys
from pathlib import Path
from source_chain import cpu_context,configure_torch_threads,require,sha256
from deit3_source import write_exact_json


def prepare(assets, source_root):
    context=cpu_context();configure_torch_threads(context)
    root=Path(assets)
    require(root.is_absolute() and '/exp-v3/' in str(root),'exp-v3 source root required')
    sys.path.insert(0,str(Path(source_root)/'runtime/bundle'))
    from retrain.efficientnet_model import (VARIANT,FORMAT,REVISION,WEIGHT_SHA,LICENSE_SHA,LIBRARY,library_identity)
    strict=root/'PUBLIC-SOURCE-STRICT.json';metadata=root/'FIXED-METADATA-RECEIPT.json'
    require(sha256(strict)=='0e88ee7eaba9a04251b51f5cf02a8ea4c363cfe47a2622c1fc1a38eb0ff81037'
            and sha256(metadata)=='4133bac75dbca9d52c0d4d73f3690617fa1eb2f0eb24d1101acd36b2222a4cc7',
            'original verified native source records differ')
    proof=json.loads(strict.read_text());weight=root/'original/pytorch_model.bin'
    license_path=root/'fixed-metadata/LICENSE-apache-2.0.txt'
    require(proof['passed'] is True and proof['execution']['uid']==context['uid']
            and proof['revision']==REVISION and proof['weight']['sha256']==WEIGHT_SHA
            and proof['public_state_tensors']==794 and proof['strict_missing_keys']==[]
            and proof['strict_unexpected_keys']==[] and proof['all_public_tensors_exact'] is True
            and proof['forward_calls']==0 and proof['GPU_initialized'] is False,
            'native strict source evidence differs')
    require(weight.is_file() and not weight.is_symlink() and weight.stat().st_size==96674637
            and sha256(weight)==WEIGHT_SHA and sha256(license_path)==LICENSE_SHA,
            'public native bytes/license differ')
    require(library_identity()==LIBRARY,'four native implementation pins differ')
    source_url=proof['weight']['url']
    provenance={'schema_version':1,'model':VARIANT,'format':FORMAT,'revision':REVISION,
        'source_url':source_url,'training_data':'ImageNet-1K','sha256':WEIGHT_SHA,
        'original_sha256':WEIGHT_SHA,'native_library':LIBRARY,'redistribution_confirmed':True,
        'license_file':str(license_path.relative_to(root)),'license_sha256':LICENSE_SHA,
        'declared_license':'Apache-2.0','full_classes':1000,'public_tensor_count':794,
        'source_configuration':proof['source_configuration'],
        'strict_record':{'path':str(strict),'sha256':sha256(strict)},
        'library_scope':'four named implementation files, not every transitive timm file',
        'source_producer_sha256':sha256(__file__)}
    path=root/'PROVENANCE.json';write_exact_json(path,provenance)
    handoff={'schema_version':1,'passed':True,'model':VARIANT,'revision':REVISION,'execution':context,
        'native_weights':{'path':str(weight),'sha256':WEIGHT_SHA},
        'provenance':{'path':str(path),'sha256':sha256(path)},'public_tensor_count':794,
        'source_strict':{'path':str(strict),'sha256':sha256(strict)},
        'forward_calls':0,'GPU_initialized':False,'datasets_read':False,
        'scope':'register existing exact public state evidence; no new model load or forward'}
    handoff_path=root/'PUBLIC-HANDOFF.json';write_exact_json(handoff_path,handoff)
    print(json.dumps({'provenance':handoff['provenance'],
        'handoff':{'path':str(handoff_path),'sha256':sha256(handoff_path)}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True);parser.add_argument('--source-root',required=True)
    args=parser.parse_args();prepare(args.assets,args.source_root)
