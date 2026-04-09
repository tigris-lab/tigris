"""Tensor lifetime computation - birth and death step for each activation."""

from tigris.graph.ir import AnalyzedGraph, TensorLifetime


def compute_lifetimes(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Walk the sorted ops and record birth/death steps for activation tensors.

    - Model inputs:  birth_step = -1
    - Model outputs: death_step = len(ops) (kept alive beyond last op)
    - Constants (weights/initializers) are skipped entirely.
    """
    num_ops = len(ag.ops)

    birth: dict[str, int] = {}
    death: dict[str, int] = {}

    # Model inputs are born before execution starts
    for name in ag.model_inputs:
        birth[name] = -1

    # Walk ops in execution order
    for op in ag.ops:
        # Outputs of this op are born at this step
        for out_name in op.outputs:
            if out_name == "":
                continue
            info = ag.tensors.get(out_name)
            if info is None or info.is_constant:
                continue
            birth[out_name] = op.step

        # Inputs consumed at this step - update last-use
        for inp_name in op.inputs:
            if inp_name == "":
                continue
            info = ag.tensors.get(inp_name)
            if info is None or info.is_constant:
                continue
            death[inp_name] = op.step

    # Model outputs must survive until after the last op
    for name in ag.model_outputs:
        death[name] = num_ops

    # Build lifetime records
    ag.lifetimes = {}
    for name in birth:
        info = ag.tensors.get(name)
        if info is None or info.is_constant:
            continue
        size = info.size_bytes
        d = death.get(name, birth[name])  # unused tensor dies at birth
        ag.lifetimes[name] = TensorLifetime(
            tensor_name=name,
            birth_step=birth[name],
            death_step=d,
            size_bytes=size,
        )

    return ag
