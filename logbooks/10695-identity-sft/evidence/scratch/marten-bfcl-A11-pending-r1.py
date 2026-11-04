import hashlib
import json
from pathlib import Path


base = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
parent = Path('/mnt/storage/karan-scratch/marten-bfcl-A10-reduced-n-pending-r1.py')
parent_receipt = json.loads((base/'bfcl-noise-r1/A10-run-bootstrap-pending-native-marten-r1.json').read_text())
assert hashlib.sha256(parent.read_bytes()).hexdigest() == parent_receipt['source_script_sha256']
source = parent.read_text().replace('A10','A11')
changes = {
    'ad360b5c06c6726c7641024092b8d8955bead17a75e5d5605f9253aa55da2f48':'66f81727ef745fd63b2decd0a32702f971b83262ee5090af97c327488946bd22',
    'A11-recovered-BFCL-five-r1/A11-run-index-20261030T195858.346212Z.json':'A11-BFCL-five-r1/A11-run-index-20261030T203747.133524Z.json',
    '285a1d199e53aaec3131827d60a0f89677157ffafec8681ec29a288f0fa16cee':'b1335baf6d6667b7c2752d611f1b9d1b9ee4ebfe824c275125156b3741e218ae',
    "    'A8a': (namespace['fractions']['A8a'],namespace['draws']['A8a']),\n":'',
    "index['complete123']==4 and index['exhausted_final_R2']==1":"index['complete123']==5 and index['exhausted_final_R2']==0",
    "assert index['ALL_CHAINS_FULFILLED_OR_EXHAUSTED'] and not index['ALL_FIVE_COMPLETE']":"assert index['ALL_CHAINS_FULFILLED_OR_EXHAUSTED'] and index['ALL_FIVE_COMPLETE']",
    "len(index['physical_runs'])==14":"len(index['physical_runs'])==8",
    'df7db2d3393d2118a189f7d93b8a4b622657df7b8552fb687d57776fe849e0ba':'da23bb682ee555674b48e1a46b3c441c1ceb9f17d78810e255ed5c8b6e738f62',
    'e3877fb0b29f537a395826923c447a65675820a864696dc1fc0935dcb1dbbb0c':'2a8f8669547e9c88c806d2fe401f1bff810dde31942964cd066bd42c95625328',
    'ac5846334f9f9de49fb7cb3bdddbac007925c3a5993ad8128755a527c20e8b7c':'2369a292724cb7112ab114cf4129b1e25307f65f9bead3244a592d0d4c15134f',
    'assert len(fractions)==4':'assert len(fractions)==5',
    'range(4)) for iteration':'range(5)) for iteration',
    'planned=5,n=4,exhausted_infra_no_score=1,pending=0,physical_workflows=14':'planned=5,n=5,exhausted_infra_no_score=0,pending=0,physical_workflows=8',
    "status='A11_COMPLETE_CHAINS_N4_STATISTICS_CONDITIONAL_ACTUAL_NATIVE_RAW_QUALIFICATION_PENDING'":"status='A11_COMPLETE_CHAINS_N5_STATISTICS_CONDITIONAL_ACTUAL_NATIVE_RAW_QUALIFICATION_PENDING'",
    'reserved_A9_fixed_n5_block_advanced_without_outcomes=True,A11_actual_n4_from_cloned_reserved_block_start=True':'reserved_A9_and_A10_fixed_n5_blocks_advanced_without_outcomes=True,A11_actual_n5_from_cloned_reserved_block_start=True',
    'A11_reserved_fixed_n5_canonical_block_advanced_for_future_A11=True':'A11_actual_n5_block_matches_original_registration=True',
    "summary=summary,contrasts=contrasts,exhausted_slot_numbers=[2]":"summary=summary,contrasts=contrasts,exhausted_slot_numbers=[]",
    "secondary=['A11_minus_A8a','A11_minus_A3']":"secondary=['A11_minus_A3','A11_minus_A9'], pending_registered_contrast=dict(A11_minus_A9='A9_LOGICAL_CHAINS_NOT_FINAL_NO_PARTIAL_MEAN_OR_IMPUTATION')",
    'actualn4 choices eachiteration':'actualn5 choices eachiteration',
    'Four evaluation clusters/one trained checkpoint':'Five evaluation clusters/one trained checkpoint',
    "A11_canonical_block_start_state_sha256=digest(json.dumps(canonical.getstate()).encode())":"after_A11_canonical_block_state_sha256=digest(json.dumps(canonical.getstate()).encode())",
}
for original,replacement in changes.items():
    assert original in source,original
    source = source.replace(original,replacement)
assert source.count('for iteration in range(10000):') == 2
source = source.replace('for iteration in range(10000):','for iteration in range(20000):',1)
exec(compile(source,str(parent),'exec'),{'__file__':__file__,'__name__':'__main__'})
