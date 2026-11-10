"""Executable catalog contract and mutation checks (CPU only)."""
import math
import random
import types

import pytest
import torch

from practice.exercises import get_exercises


def run_case(exercise, case, source=None):
    # A fresh module, RNG context, and namespace for every test: no case can leak state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(90210)
        rng_state = random.getstate()
        try:
            random.seed(90210)
            module = types.ModuleType("candidate")
            exec(compile(source or exercise["reference"], "<candidate>", "exec"), module.__dict__)
            exec(compile(case["code"], "<test>", "exec"), {
                "candidate": module, "torch": torch, "math": math, "random": random,
            })
        finally:
            random.setstate(rng_state)


def test_catalog_contract_and_privacy():
    private = get_exercises(include_private=True)
    public = get_exercises()
    assert len(private) == len(public) == 20
    assert [e["number"] for e in private] == list(range(82, 102))
    for e, p in zip(private, public):
        assert e["id"] == f'C{e["number"]}'
        assert e["source_id"] == f'Q{e["number"]}'
        assert e["chapter"] == "代码"
        assert e["difficulty"] in ("基础", "进阶")
        assert all(e.get(k) for k in ("title", "description", "starter", "hints", "reference"))
        assert "reference" not in p and "tests" not in p
        assert len(e["tests"]) >= 3
        assert len({t["name"] for t in e["tests"]}) == len(e["tests"])
        compile(e["starter"], "<starter>", "exec")
    private[0]["hints"].append("mutation")
    assert "mutation" not in get_exercises(True)[0]["hints"]


@pytest.mark.parametrize("number", range(82, 102))
def test_reference_passes_all_cases(number):
    e = get_exercises(True)[number - 82]
    for case in e["tests"]:
        try:
            run_case(e, case)
        except Exception as exc:
            raise AssertionError(f'{e["id"]}: {case["name"]}: {exc}') from exc


@pytest.mark.parametrize("number", range(82, 102))
def test_unimplemented_starter_is_rejected(number):
    e = get_exercises(True)[number - 82]
    failures = 0
    for case in e["tests"]:
        try:
            run_case(e, case, e["starter"])
        except Exception:
            failures += 1
    assert failures > 0, f'{e["id"]} tests must reject the unfinished starter'


@pytest.mark.parametrize("number,old,new", [
    (82, "unbiased=False", "unbiased=True"),
    (87, "math.sqrt(dh)", "math.sqrt(d)"),
    (88, "z=layer_norm(x,(d,),eps=eps)", "z=x"),
    (92, "min(ii+block_size,m)", "min(ii+block_size-1,m)"),
    (95, "((x+1,y),(x-1,y),(x,y+1),(x,y-1))", "((x+1,y),(x-1,y),(x,y+1),(x,y-1),(x+1,y+1))"),
    (96, "row.reverse()", "pass"),
    (98, "cols=len(matrix[0])", "cols=len(matrix)"),
    (99, "max(1.0,float(x))", "float(x)"),
    (83, "x = x + dt * model(x, t)", "x = x - dt * model(x, t)"),
    (84, "delta + gamma * lam * (1 - dones[t]) * carry", "delta + gamma * lam * carry"),
    (85, "unbiased=False", "unbiased=True"),
    (86, "scores.masked_fill(~mask, float('-inf'))", "scores"),
    (89, "torch.logsumexp(logits, dim=-1)", "torch.log(torch.exp(logits).sum(dim=-1))"),
    (90, "offset + torch.arange(q.shape[1]", "0 * offset + torch.arange(q.shape[1]"),
    (91, "cache['version'] == version", "True"),
    (93, "reverse=True", "reverse=False"),
    (94, "water += left_max - heights[left]", "water += heights[left] - left_max"),
    (97, "while slow is not fast:", "while slow.val != fast.val:"),
    (100, "nums[maximum[0]] - nums[minimum[0]] > limit", "nums[maximum[0]] - nums[minimum[0]] >= limit"),
    (101, "if prefix not in first:", "if True:"),
])
def test_typical_wrong_implementations_are_rejected(number, old, new):
    e = get_exercises(True)[number - 82]
    assert old in e["reference"], "mutation target drifted"
    broken = e["reference"].replace(old, new)
    failures = 0
    for case in e["tests"]:
        try:
            run_case(e, case, broken)
        except Exception:
            failures += 1
    assert failures, f'C{number} typical bug was not detected'
