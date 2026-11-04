import hashlib
import json
from pathlib import Path


base = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
parent = Path('/mnt/storage/karan-scratch/marten-bfcl-A10-reduced-n-pending-r1.py')
parent_receipt = json.loads((base/'bfcl-noise-r1/A10-run-bootstrap-pending-native-marten-r1.json').read_text())
assert hashlib.sha256(parent.read_bytes()).hexdigest() == parent_receipt['source_script_sha256']
source = parent.read_text().replace('A10','A9')
changes = {
    'ad360b5c06c6726c7641024092b8d8955bead17a75e5d5605f9253aa55da2f48':'0aa09019d739a0d106d11acb09334dc9b2827200f192573f9f5e89237cc9aa77',
    'A9-recovered-BFCL-five-r1/A9-run-index-20261030T195858.346212Z.json':'A9-recovered-BFCL-five-r1/A9-run-index-20261030T204949.679676Z.json',
    '285a1d199e53aaec3131827d60a0f89677157ffafec8681ec29a288f0fa16cee':'9d41e6ebe03f3466153d5a920bd6c3bc31e9daca41b6a7c82eccd1f34cb7da9b',
    "for iteration in range(10000):\n    for cluster in range(5):\n        canonical.choice(range(5))\na10_start = canonical.getstate()":"a10_start = canonical.getstate()",
    "index['complete123']==4 and index['exhausted_final_R2']==1":"index['complete123']==5 and index['exhausted_final_R2']==0",
    "assert index['ALL_CHAINS_FULFILLED_OR_EXHAUSTED'] and not index['ALL_FIVE_COMPLETE']":"assert index['ALL_CHAINS_FULFILLED_OR_EXHAUSTED'] and index['ALL_FIVE_COMPLETE']",
    "len(index['physical_runs'])==14":"len(index['physical_runs'])==12",
    'df7db2d3393d2118a189f7d93b8a4b622657df7b8552fb687d57776fe849e0ba':'ecd3e7968d80e4bfc0c8bbf841919672f4dd999524c3aa197d1c0222ecb5518e',
    'e3877fb0b29f537a395826923c447a65675820a864696dc1fc0935dcb1dbbb0c':'1a277895930db529b04ab941fda5b05d2aea398ec1e5f52b76fd96601e7773af',
    'ac5846334f9f9de49fb7cb3bdddbac007925c3a5993ad8128755a527c20e8b7c':'51c0e88a9e4dbf018f1c21dcd68c2807bc3d5a1bbaee77d1831b31c8e75760aa',
    'assert len(fractions)==4':'assert len(fractions)==5',
    'range(4)) for iteration':'range(5)) for iteration',
    'planned=5,n=4,exhausted_infra_no_score=1,pending=0,physical_workflows=14':'planned=5,n=5,exhausted_infra_no_score=0,pending=0,physical_workflows=12',
    "status='A9_COMPLETE_CHAINS_N4_STATISTICS_CONDITIONAL_ACTUAL_NATIVE_RAW_QUALIFICATION_PENDING'":"status='A9_COMPLETE_CHAINS_N5_STATISTICS_CONDITIONAL_ACTUAL_NATIVE_RAW_QUALIFICATION_PENDING'",
    'reserved_A9_fixed_n5_block_advanced_without_outcomes=True,A9_actual_n4_from_cloned_reserved_block_start=True':'frozen_A8_prefix_ends_at_A9_reserved_start=True,A9_actual_n5_from_cloned_reserved_block_start=True',
    'A9_reserved_fixed_n5_canonical_block_advanced_for_future_A11=True':'A9_reserved_fixed_n5_canonical_block_advanced_for_future_A10=True',
    'A11_canonical_block_start_state_sha256':'A10_canonical_block_start_state_sha256',
    "summary=summary,contrasts=contrasts,exhausted_slot_numbers=[2]":"summary=summary,contrasts=contrasts,exhausted_slot_numbers=[]",
    'actualn4 choices eachiteration':'actualn5 choices eachiteration',
    'Four evaluation clusters/one trained checkpoint':'Five evaluation clusters/one trained checkpoint',
}
for original,replacement in changes.items():
    assert original in source,original
    source = source.replace(original,replacement)
exec(compile(source,str(parent),'exec'),{'__file__':__file__,'__name__':'__main__'})
