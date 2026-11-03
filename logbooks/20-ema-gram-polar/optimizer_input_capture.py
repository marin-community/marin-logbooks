# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose existing polar-normalization operands by extending full-update outputs."""

import jax
from optimizer_cost import NAMES


def capture_program(fn, exemplars, capture_kind="qkv_inputs"):
    """Keep every traced equation and append selected existing normalization operands.

    This modifies outputs, not optimizer equations. Evaluation and extra outputs
    can still change compilation; compare both controls with the direct function.
    """
    assert capture_kind in ("qkv_inputs", "q_norm")
    closed, output_shape = jax.make_jaxpr(fn, return_shape=True)(*exemplars)
    program = closed.jaxpr
    output_tree = jax.tree.structure(output_shape)
    assert len(program.outvars) == 28
    producers = {v: index for index, eqn in enumerate(program.eqns) for v in eqn.outvars}
    normalization = {
        index: eqn
        for index, eqn in enumerate(program.eqns)
        if eqn.primitive.name == "div" and str(eqn.source_info.name_stack).endswith("gram_direction/vmap()")
    }
    assert len(normalization) == len(NAMES), "Expected one polar normalization per matrix group"
    mapping, captured = {}, []
    selected = set()
    for group, final in zip(NAMES, program.outvars[:7], strict=True):
        pending, ancestors = [final], set()
        while pending:
            value = pending.pop()
            if isinstance(value, jax.extend.core.Literal):
                continue
            index = producers.get(value)
            if index is None or index in ancestors:
                continue
            ancestors.add(index)
            pending.extend(program.eqns[index].invars)
        candidates = ancestors & normalization.keys()
        assert len(candidates) == 1, (group, candidates)
        index = candidates.pop()
        assert index not in selected
        selected.add(index)
        eqn = normalization[index]
        inner, denominator = eqn.invars
        assert len(inner.aval.shape) == 3 and denominator.aval.shape == (inner.aval.shape[0], 1, 1)
        mapping[group] = dict(
            equation_index=index,
            source_scope=str(eqn.source_info.name_stack),
            inner_shape=list(inner.aval.shape),
            denominator_shape=list(denominator.aval.shape),
            dtype=str(inner.aval.dtype),
            role="Division numerator is pre-normalization inner; denominator is broadcast(norm + eps)",
        )
        if capture_kind == "qkv_inputs" and group in NAMES[:3]:
            captured.extend((inner, denominator))
        elif capture_kind == "q_norm" and group == "q_proj":
            captured.append(denominator)
    extended = program.replace(outvars=[*program.outvars, *captured])
    assert all(a is b for a, b in zip(program.eqns, extended.eqns, strict=True))
    assert extended.invars == program.invars and extended.constvars == program.constvars
    assert extended.outvars[:28] == program.outvars

    def evaluate(bank):
        leaves = jax.core.eval_jaxpr(program, closed.consts, *jax.tree.leaves(bank))
        return jax.tree.unflatten(output_tree, leaves)

    def capture(bank):
        leaves = jax.core.eval_jaxpr(extended, closed.consts, *jax.tree.leaves(bank))
        full = jax.tree.unflatten(output_tree, leaves[:28])
        values = (
            (leaves[28],) if capture_kind == "q_norm" else tuple((leaves[i], leaves[i + 1]) for i in range(28, 34, 2))
        )
        return full, values

    return (
        evaluate,
        capture,
        dict(
            mapping=mapping,
            original_equations=len(program.eqns),
            extended_equations=len(extended.eqns),
            original_output_leaves=28,
            capture_kind=capture_kind,
            capture_output_leaves=len(extended.outvars),
            appended_output_leaves=len(captured),
            equations_preserved_by_identity=True,
            input_and_constant_variables_preserved=True,
        ),
        str(closed),
    )
