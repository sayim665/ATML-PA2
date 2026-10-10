"""Reproduce the objective-validation evidence for Tasks 1-3.

For each task we evaluate the CURRENT (corrected) objective and, for comparison, the original defective
formula from the starter release, re-implemented inline so the defect is documented reproducibly.
Run: python -m scripts.validate_objectives
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from common.logging_utils import save_json
from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_policy_loss
from task3_grpo.grpo import group_relative_advantages


def dpo_checks():
    pc, pr = torch.tensor([-10.0]), torch.tensor([-30.0])
    l1, _ = dpo_loss(torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.zeros(4), 0.1)
    l2, _ = dpo_loss(pc, pr, pc.clone(), pr.clone(), 0.1)
    l3, d3 = dpo_loss(pc + 5, pr, pc, pr, 0.1)
    buggy_l2 = -F.logsigmoid(0.1 * ((pc - pr) + (pc - pr))).mean()        # starter: policy_margin + ref_margin
    return {
        "bug": "logits used policy_margin + ref_margin; manual requires policy_margin - ref_margin",
        "expected_init_loss": math.log(2),
        "check1_policy_eq_ref_zeros": l1.item(),
        "check2_policy_eq_ref_ref_prefers_chosen": l2.item(),
        "check2_with_original_defect": buggy_l2.item(),
        "check3_policy_improved": l3.item(),
        "check3_accuracy": d3["preference_accuracy"].item(),
    }


def ppo_checks():
    A = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    ratio = torch.tensor([[1.5, 0.5, 0.5, 1.5]])
    mask = torch.ones(1, 4)

    def run(objective_fn):
        new = torch.log(ratio).clone().requires_grad_(True)
        r = torch.exp(new)
        obj = objective_fn(r * A, r.clamp(0.8, 1.2) * A)
        (-(obj * mask).sum() / mask.sum()).backward()
        return {"objective_sum": round(obj.sum().item(), 4), "grad_nonzero": [bool(abs(g) > 1e-8) for g in new.grad[0].tolist()]}

    new = torch.log(ratio).clone().requires_grad_(True)
    loss, _, cf = ppo_policy_loss(new, torch.zeros(1, 4), A, mask, eps=0.2)
    loss.backward()
    adv, _ = compute_gae(torch.tensor([[0.0, 0.0, 1.0, 0.0]]), torch.zeros(1, 4), torch.tensor([[1.0, 1.0, 1.0, 0.0]]), 1.0, 1.0)
    return {
        "bug": "clipped surrogate used torch.maximum; manual requires min(rho*A, clip(rho)*A)",
        "cases": "(A, ratio) = (+1,1.5) (+1,0.5) (-1,0.5) (-1,1.5), eps=0.2",
        "current_objective": {"objective_sum": round(-loss.item() * 4, 4),
                              "grad_nonzero": [bool(abs(g) > 1e-8) for g in new.grad[0].tolist()],
                              "clip_fraction": cf.item()},
        "expected": {"objective_sum": -0.6, "grad_nonzero": [False, True, False, True]},
        "with_original_defect": run(torch.maximum),
        "gae": {"advantages": adv[0].tolist(), "expected": [1.0, 1.0, 1.0, 0.0]},
    }


def grpo_checks():
    r, gid = torch.tensor([9.0, 11.0, -1.0, 1.0]), torch.tensor([0, 0, 1, 1])
    flat_r = torch.tensor([3.0, 3.0, 5.0, 7.0])
    original = lambda x: (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)   # starter: whole-batch stats
    return {
        "bug": "group_relative_advantages normalized over the whole batch and ignored group_ids",
        "note": "with prompts_per_update=1 the batch is one group, so the defect is invisible in the standard "
                "continuation but corrupts any multi-group batch (e.g. the K regrouping study)",
        "current": {"advantages": [round(x, 4) for x in group_relative_advantages(r, gid).tolist()],
                    "flat_group_advantages": [round(x, 4) for x in group_relative_advantages(flat_r, gid).tolist()]},
        "expected": {"advantages": [-1.0, 1.0, -1.0, 1.0], "flat_group_advantages": [0.0, 0.0, -1.0, 1.0]},
        "with_original_defect": {"advantages": [round(x, 4) for x in original(r).tolist()],
                                 "flat_group_advantages": [round(x, 4) for x in original(flat_r).tolist()]},
    }


def main():
    out = {"task1_dpo": dpo_checks(), "task2_ppo": ppo_checks(), "task3_grpo": grpo_checks()}
    save_json("results/task1_dpo/dpo_loss_validation.json", out["task1_dpo"])
    save_json("results/task2_ppo/ppo_objective_validation.json", out["task2_ppo"])
    save_json("results/task3_grpo/grpo_objective_validation.json", out["task3_grpo"])
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
