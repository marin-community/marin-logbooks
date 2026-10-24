import hashlib
import json
import re
from pathlib import Path

BASE=Path('/private/tmp/native-jax-bf16-splash-15b-graph-context-results')

def parse(name):
    source=(BASE/(name+'.hlo.txt')).read_text().splitlines()
    blocks={}; block=None
    for number,line in enumerate(source,1):
        if re.match(r'(?:ENTRY )?%[\w.-]+ .*\{\s*$',line):
            block=re.search(r'%([\w.-]+)',line).group(1); blocks[block]={'instructions':{},'line':number}
        elif line=='}': block=None
        elif block is not None:
            match=re.match(r'\s*(ROOT )?(%[\w.-]+) = (.*)',line)
            if not match: continue
            root,key,body=match.groups()
            clean=re.split(r', (?:frontend_attributes|metadata|backend_config)=',body)[0]
            parsed=re.match(r'(.*) ([\w-]+)\(([^)]*)\)(.*)',clean)
            if not parsed: raise ValueError((name,number,clean[:200]))
            shape,op,args,attrs=parsed.groups()
            blocks[block]['instructions'][key]={'shape':re.sub(r'\{[^{}]*\}','',shape),'op':op,'args':args,'attrs':attrs,'line':number}
            if root: blocks[block]['root']=key
    return blocks

def canonical(blocks,block_name,inputs=None):
    block=blocks[block_name]; memo={}
    def node(key):
        if key in memo:return memo[key]
        inst=block['instructions'][key]; op=inst['op']
        if op=='parameter': result=('arg',int(inst['args'])) if inputs is None else inputs[int(inst['args'])]
        elif op=='fusion':
            args=[node(v) for v in re.findall(r'%[\w.-]+',inst['args'])]
            callee=re.search(r'calls=%([\w.-]+)',inst['attrs']).group(1)
            result=canonical(blocks,callee,args)
        else:
            args=tuple(node(v) for v in re.findall(r'%[\w.-]+',inst['args']))
            attrs=inst['attrs']
            callee=re.search(r'to_apply=%([\w.-]+)',attrs)
            if callee:
                attrs=attrs[:callee.start()]+str(canonical(blocks,callee.group(1)))+attrs[callee.end():]
            result=(op,inst['shape'],inst['args'] if op in ('constant','iota') else args,attrs)
        memo[key]=result;return result
    return node(block['root'])

def digest(value):return hashlib.sha256(repr(value).encode()).hexdigest()

A=parse('A-forward-combined'); B=parse('B-full-alone'); C=parse('C-composed-alone')
pairs=[('gate','fused_computation.24','fused_computation.25','fused_computation.18'),('silu_up','fused_computation.13','fused_computation.12','fused_computation.8'),('down','fused_computation.39','fused_computation.37','fused_computation.29')]
# Standalone names discovered by its actual entry calls, not assumed here.
checks=[]
for stage,full,manual,_ in pairs:
    x=canonical(A,full);y=canonical(A,manual)
    checks.append({'stage':stage,'full':full,'manual':manual,'full_line':A[full]['line'],'manual_line':A[manual]['line'],'equivalent':x==y,'full_sha256':digest(x),'manual_sha256':digest(y)})
# Locate every single-output standalone function with the same tail semantic signature.
for check,(_,full,manual,_) in zip(checks,pairs):
    target=canonical(A,full)
    for label,blocks in [('B',B),('C',C)]:
        matches=[]
        for name,block in blocks.items():
            if 'root' not in block or not name.startswith('fused_computation'):continue
            try: value=canonical(blocks,name)
            except (KeyError,AttributeError):continue
            if value==target:matches.append({'function':name,'line':block['line']})
        check[label+'_matching_functions']=matches
report={'scope':'Optimized HLO tail arithmetic comparison. Inline fusion subcomputations, alpha-normalize SSA, ignore only instruction metadata/frontend/backend configs and physical layout/memory-space suffixes. Preserve dtypes, logical shapes, opcode, ordering, constants, convolution attributes and reduction bodies. Parameters are positional symbols; equivalence is conditional on equal corresponding runtime operands. This is not execution/codegen validation.','comparisons':checks}
Path('/private/tmp/native-jax-bf16-splash-15b-optimized-tail-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
