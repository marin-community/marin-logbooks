import hashlib
from pathlib import Path


parent = Path('/mnt/storage/karan-scratch/marten-A3-paired-maths-pending-r1.py')
assert hashlib.sha256(parent.read_bytes()).hexdigest() == '06d9c4d18ce3a5cf179b7fa2bb2f8b9c574cd2faff8f65586a4976bdb8f8d5bd'
source = parent.read_text()
assert source.count("ROOT = BASE / 'A3-paired-math-infra-replacement1-r1'") == 1
source = source.replace("ROOT = BASE / 'A3-paired-math-infra-replacement1-r1'", "ROOT = BASE / 'A11-paired-math-r1'")
source = source.replace('A3', 'A11')
assert source.count('A11-A11-paired-maths-preregistration-marten-r1.json') == 1
source = source.replace('A11-A11-paired-maths-preregistration-marten-r1.json', 'A3-A11-paired-maths-preregistration-marten-r1.json')
assert source.count('824aa6b52dbb36c947b7cd6c0728ac8e06bd2ba42a2db2683a738089b347ee7f') == 1
source = source.replace('824aa6b52dbb36c947b7cd6c0728ac8e06bd2ba42a2db2683a738089b347ee7f', 'e59c02c330f826c91815850fd0f9ca188cc67d53f554ab6c839d935773cf8a97')
old = 'First zero-output preempted A11 physical attempt remains separate, no preferred-success selection'
assert source.count(old) == 1
source = source.replace(old, 'One original A11 candidate maths workflow; no new reference or preferred-success selection')
exec(compile(source, str(parent), 'exec'), {'__file__': __file__, '__name__': '__main__'})
