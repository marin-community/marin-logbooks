# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose existing polar-normalization operands by extending full-update outputs."""

import jax
import numpy as np
from optimizer_cost import NAMES


def capture_program(fn, exemplars, capture_kind="qkv_inputs"):
    """Keep every traced equation and append selected existing normalization operands.

    This modifies outputs, not optimizer equations. Evaluation and extra outputs
    can still change compilation; compare both controls with the direct function.
    """
    assert capture_kind in ("qkv_inputs", "q_norm", "q_inner", "q_pair")
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
        elif capture_kind in ("q_norm", "q_inner") and group == "q_proj":
            captured.append(denominator if capture_kind == "q_norm" else inner)
        elif capture_kind == "q_pair" and group == "q_proj":
            captured.extend((inner, denominator))
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
            tuple(leaves[28:])
            if capture_kind in ("q_norm", "q_inner", "q_pair")
            else tuple((leaves[i], leaves[i + 1]) for i in range(28, 34, 2))
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


def replace_division_operands(program, index):
    """Replace two uses at one division; all other uses retain their variables."""
    assert 0 <= index < len(program.eqns), "Missing target division"
    target = program.eqns[index]
    assert target.primitive.name == "div" and len(target.invars) == 2
    external = [jax.extend.core.Var(value.aval) for value in target.invars]
    replacement = target.replace(invars=external)
    equations = list(program.eqns)
    equations[index] = replacement
    modified = program.replace(invars=[*program.invars, *external], eqns=equations)
    assert modified.outvars == program.outvars and modified.constvars == program.constvars
    assert modified.invars[:-2] == program.invars
    assert replacement.primitive is target.primitive and replacement.params == target.params
    assert replacement.outvars == target.outvars and replacement.effects == target.effects
    assert all(a is b for i, (a, b) in enumerate(zip(program.eqns, modified.eqns, strict=True)) if i != index)
    return modified


def substitution_program(fn, exemplars):
    """Use the capture mapper, then modify only q_proj's division operand uses."""
    _, _, mapping, source = capture_program(fn, exemplars, "q_pair")
    closed, output_shape = jax.make_jaxpr(fn, return_shape=True)(*exemplars)
    assert str(closed) == source, "Capture mapping and substitution trace disagree"
    index = mapping["mapping"]["q_proj"]["equation_index"]
    modified = replace_division_operands(closed.jaxpr, index)
    output_tree = jax.tree.structure(output_shape)

    def evaluate(bank):
        inputs, pair = bank
        leaves = jax.core.eval_jaxpr(modified, closed.consts, *jax.tree.leaves(inputs), *pair)
        return jax.tree.unflatten(output_tree, leaves)

    return (
        evaluate,
        dict(
            target=mapping["mapping"]["q_proj"],
            original_equations=len(closed.jaxpr.eqns),
            modified_equations=len(modified.eqns),
            original_output_leaves=len(modified.outvars),
            appended_input_leaves=2,
            only_target_operand_uses_replaced=True,
            all_other_equations_preserved_by_identity=True,
            outputs_constants_original_inputs_preserved=True,
            original_internal_identity_established=False,
        ),
        str(modified),
    )


def substitution_fixtures():
    """Small CPU controls, called only after the harness opens online tracking."""
    shape = jax.ShapeDtypeStruct((2,), np.float32)
    closed = jax.make_jaxpr(lambda x: (x / x, x + x))(shape)
    index = next(i for i, e in enumerate(closed.jaxpr.eqns) if e.primitive.name == "div")
    modified = replace_division_operands(closed.jaxpr, index)
    x = np.array([2, 4], dtype=np.float32)
    numerator = np.array([6, 12], dtype=np.float32)
    denominator = np.array([3, 4], dtype=np.float32)
    actual = jax.jit(lambda a, b, c: jax.core.eval_jaxpr(modified, closed.consts, a, b, c))(x, numerator, denominator)
    assert np.array_equal(actual[0], numerator / denominator)
    assert np.array_equal(actual[1], x + x), "Another use of the original variable changed"
    for bad in (-1, len(closed.jaxpr.eqns), 1):
        try:
            replace_division_operands(closed.jaxpr, bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid target was accepted")

    def missing(x):
        return tuple(x for _ in range(28))

    def ambiguous(x):
        with jax.named_scope("gram_direction/vmap()"):
            values = [x / (float(i) + 1) for i in range(7)]
        return tuple([values[0] + values[1], *values[1:], *([x] * 21)])

    for fn in (missing, ambiguous):
        try:
            capture_program(fn, (shape,), "q_pair")
        except AssertionError as error:
            if fn is missing:
                assert "Expected one polar normalization" in str(error)
            else:
                assert error.args[0][0] == "q_proj" and len(error.args[0][1]) == 2
        else:
            raise AssertionError("Missing or ambiguous mapping was accepted")
    return dict(
        repeated_operand_positions=True,
        non_target_uses_unchanged=True,
        output_order_shapes_preserved=True,
        invalid_target_rejected=True,
        missing_ambiguous_mapping_rejected=True,
        numerical_fixture_bitwise_equal=True,
    )
