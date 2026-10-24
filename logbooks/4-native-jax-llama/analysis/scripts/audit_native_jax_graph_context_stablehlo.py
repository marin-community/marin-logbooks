"""Read-only data-flow fingerprints for the retained StableHLO forward roots."""
from pathlib import Path
import collections
import hashlib
import json
import re
from jax._src.interpreters import mlir
from jaxlib.mlir import ir

ROOT = Path('/private/tmp/native-jax-bf16-splash-15b-graph-context-results')
OUTPUT = Path('/private/tmp/native-jax-bf16-splash-15b-graph-context-stablehlo-audit.json')

def sha(data):
    return hashlib.sha256(data).hexdigest()

class Graph:
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.functions = {ir.StringAttr(f.attributes['sym_name']).value: f for f in module.body.operations if f.operation.name == 'func.func'}
        self.block = self.functions['main'].regions[0].blocks[0]
        self.values = {}
        self.operations = {}
        self.functions_cache = {}
        self.nodes = {}
        arg_attrs = self.functions['main'].attributes['arg_attrs']
        for i, value in enumerate(self.block.arguments):
            if name == 'D-original-capture':
                label = f'arg{i}' if i < 9 else ('arg9' if i == 11 else f'D-extra{i}')
            else:
                label = f'arg{i}'
            self.values[value] = ['main_argument', label, str(value.type), str(arg_attrs[i])]
        self.returns = list(list(self.block.operations)[-1].operands)

    def function(self, name):
        if name not in self.functions_cache:
            fn = self.functions[name]
            self.functions_cache[name] = self.region(fn.regions[0])
        return self.functions_cache[name]

    def region(self, region):
        blocks = []
        for block in region.blocks:
            for i, value in enumerate(block.arguments):
                self.values[value] = ['region_argument', i, str(value.type)]
            blocks.append({'arguments': [str(v.type) for v in block.arguments], 'operations': [self.operation(op.operation) for op in block.operations]})
        return blocks

    def operation(self, op):
        op = op.operation
        if op in self.operations:
            return self.operations[op]
        attributes = {}
        for attr_name, attr_value in op.attributes.items():
            if attr_name == 'callee':
                attributes['callee_body'] = self.function(ir.FlatSymbolRefAttr(attr_value).value)
            else:
                attributes[attr_name] = str(attr_value)
        result = {'opcode': op.name, 'results': [str(v.type) for v in op.results],
            'attributes': attributes, 'operands': [self.value(v) for v in op.operands],
            'regions': [self.region(region) for region in op.regions]}
        fingerprint = sha(json.dumps(result, sort_keys=True, separators=(',', ':')).encode())
        self.operations[op] = fingerprint
        self.nodes[fingerprint] = result
        return fingerprint

    def value(self, value):
        if value in self.values:
            return self.values[value]
        op = value.owner
        result_number = next(i for i, v in enumerate(op.results) if v == value)
        result = ['operation_result', self.operation(op), result_number, str(value.type)]
        self.values[value] = result
        return result

    def root(self, index):
        return self.value(self.returns[index])


def first_diff(a, av, b, bv, path='root'):
    if av == bv:
        return None
    if isinstance(av, list) and isinstance(bv, list) and av and bv and av[0] == bv[0] == 'operation_result':
        an, bn = a.nodes[av[1]], b.nodes[bv[1]]
        for key in ('opcode', 'results', 'attributes', 'regions'):
            if an[key] != bn[key]:
                return {'path': path, 'kind': key, 'left': str(an[key])[:1500], 'right': str(bn[key])[:1500]}
        if len(an['operands']) != len(bn['operands']):
            return {'path': path, 'kind': 'operand_count'}
        for i, (aa, bb) in enumerate(zip(an['operands'], bn['operands'], strict=True)):
            found = first_diff(a, aa, b, bb, path + f'/{an["opcode"]}[{i}]')
            if found:
                return found
    return {'path': path, 'kind': 'value', 'left': av, 'right': bv}

with mlir.make_ir_context():
    graphs = {name: Graph(ir.Module.parse((ROOT/f'{name}.stablehlo.txt').read_text()), name) for name in ('A-forward-combined','B-full-alone','C-composed-alone','D-original-capture')}
    roots = {'A_residual': ('A-forward-combined',1),'A_full':('A-forward-combined',2),'A_composed':('A-forward-combined',3),'B_full':('B-full-alone',1),'C_composed':('C-composed-alone',1),'D_residual':('D-original-capture',1),'D_full':('D-original-capture',2),'D_composed':('D-original-capture',3)}
    root_values = {name: graphs[case].root(index) for name,(case,index) in roots.items()}
    pairs = [('B_full','C_composed'),('A_full','B_full'),('A_composed','C_composed'),('A_full','A_composed'),('D_full','A_full'),('D_composed','A_composed'),('D_residual','A_residual')]
    comparisons=[]
    for left,right in pairs:
        a,b=graphs[roots[left][0]],graphs[roots[right][0]]
        comparisons.append({'left':left,'right':right,'dataflow_equal':root_values[left]==root_values[right], 'first_difference':first_diff(a,root_values[left],b,root_values[right])})
    report={'schema':'native-jax-graph-context-stablehlo-dataflow-audit.v1','scope':'Read-only MLIR parsing, no source changes, compilation, execution, hardware, job query or downloads.','normalization':'Structural DAG fingerprints include opcode, result types, all operation attributes, literal values, operand dependencies, nested regions, and called function bodies. SSA/function names and debug locations are excluded. Main argument sharding attributes are included; module replicas/partitions and mesh declarations are separately checked. D model inputs0..8 match A/B/C; D arg11 is the same input x mapped to A/B/C arg9. Extra D suffix arguments remain distinct. This is structural equality, not a general algebraic-equivalence proof.','script_sha256':sha(Path(__file__).read_bytes()),'stablehlo_file_sha256':{name:sha((ROOT/f'{name}.stablehlo.txt').read_bytes()) for name in graphs},'root_fingerprints':root_values,'comparisons':comparisons}
    module_attrs = {name: {k:str(v) for k,v in graph.module.operation.attributes.items() if k in ('mhlo.num_partitions','mhlo.num_replicas')} for name,graph in graphs.items()}
    meshes = {name: [str(op) for op in graph.module.body.operations if op.operation.name == 'sdy.mesh' and str(op.attributes['sym_name']) == '"mesh"'] for name,graph in graphs.items()}
    assert all(value == {'mhlo.num_partitions':'4 : i32','mhlo.num_replicas':'1 : i32'} for value in module_attrs.values())
    assert all(meshes[name] == meshes['A-forward-combined'] for name in meshes)
    report['module_replica_partition_attributes'] = module_attrs
    report['common_main_mesh_declaration'] = meshes['A-forward-combined']
    report['module_meshes_verified_equal'] = True
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))
