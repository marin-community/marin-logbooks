import hashlib
from pathlib import Path


parent = Path('/mnt/storage/karan-scratch/marten-M51-A12-paired-maths-final-r1.py')
source = parent.read_text()
assert hashlib.sha256(parent.read_bytes()).hexdigest() == 'f797e11b102482163fd4113df606c3704549e2a8c3de5d7843104c004b05cfc1'
targets = {
    'A13': {'index': '25cb90509f77c209bc9b176cbc61bde00904e20352fb139294f67ffe39271e59', 'runtime': '3200d53eaa2330458742738eb10e391387201b9a44a7f33217949f8804941eee', 'binding': '94778c89119e6c2614fbd7f41d7142d3ebaa9a9fcb1e6296dcd1e5ff8eaf0ca0', 'science': 'd3c966d90391dd7e1fabb5e6e29ecdb3382f454c71fd3a4a312878d03cea8738'},
    'M25': {'index': 'b1afd3584e1e36d399c17a98b1268a00b45e3382be6a46fa31f493ae25b260fe', 'runtime': 'd2a601eb637631833d056486c522ad461c0416af2e3f5fd2559bbadf86620c72', 'binding': 'ba5ade29bc74ee5996c52e2824275c3e4cfe4424d5a73412fc0300aa87805750', 'science': '108c2f65840ea7ab22acdb677494726a35d316e7400af5ba048337e84d44c4b9'},
    'M49': {'index': '6523437acbbf02cfefb8454f087610cbfc9437a8a8d5148f8820d5a0b033e165', 'runtime': '38c243d439945114c755d4a6cd3d98eeeef3bced9acefc6c95d88de407f463b8', 'binding': '5675e0e4dc2e9b46182e13831b739e1398892d6d26f3e62c10e2c60ab9d46ff0', 'science': '108c2f65840ea7ab22acdb677494726a35d316e7400af5ba048337e84d44c4b9'},
    'M75': {'index': '62b5ca76db72c3dcc459206b474048ef22c955c7cdf8123f20d640cd3c38859c', 'runtime': '6a53222897b46d9fdc8b208fd9e2c490c08a9587b6217ba93ab5e6255129d69e', 'binding': '1fde9cd378d7127abaeeba3f1f7faf8aeb97edac3d0d1fd93c65b59694352725', 'science': '108c2f65840ea7ab22acdb677494726a35d316e7400af5ba048337e84d44c4b9'},
}
start = source.index('TARGETS = {')
end = source.index('\n\n\ndef digest', start)
source = source[:start] + 'TARGETS = ' + repr(targets) + source[end:]
before = "    assert digest(root / (label + '-candidate.json')) == target['candidate_sha256']"
after = "    committed = Path(plan['workspace']) / target['committed_path']\n    assert digest(committed) == target['candidate_sha256']\n    assert json.loads((root / (label + '-candidate.json')).read_text()) == json.loads(committed.read_text()) == target['candidate']"
assert source.count(before) == 1
source = source.replace(before, after)
exec(compile(source, str(parent), 'exec'), {'__file__': __file__, '__name__': '__main__'})
