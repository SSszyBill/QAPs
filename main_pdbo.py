import argparse
import os
import time

import numpy as np
import torch

import main_gpu3 as srpd


def project_simplex_rows(V):
    """Euclidean projection of each row of V onto the probability simplex."""
    orig_shape = V.shape
    n = orig_shape[-1]
    flat = V.reshape(-1, n)
    U, _ = torch.sort(flat, dim=1, descending=True)
    cssv = torch.cumsum(U, dim=1) - 1.0
    ind = torch.arange(1, n + 1, device=V.device, dtype=V.dtype).view(1, n)
    cond = U - cssv / ind > 0
    rho = cond.sum(dim=1).clamp_min(1) - 1
    theta = cssv.gather(1, rho.view(-1, 1)) / (rho.to(V.dtype).view(-1, 1) + 1.0)
    return torch.clamp(flat - theta, min=0.0).reshape(orig_shape)


def project_birkhoff(P, num_iters):
    """Alternating row/column simplex projections onto the Birkhoff polytope."""
    Z = P
    for _ in range(num_iters):
        Z = project_simplex_rows(Z)
        Z = project_simplex_rows(Z.transpose(-1, -2)).transpose(-1, -2)
    return project_simplex_rows(Z)


def project_birkhoff_normalize(P, num_iters, eps=1e-12):
    """Fast primal projection by alternating nonnegative row/column scaling."""
    Z = torch.clamp(P, min=0.0)
    for _ in range(num_iters):
        Z = Z / Z.sum(dim=-1, keepdim=True).clamp_min(eps)
        Z = Z / Z.sum(dim=-2, keepdim=True).clamp_min(eps)
    return Z


def project_primal(P, num_iters, method):
    if method == "simplex":
        return project_birkhoff(P, num_iters)
    if method == "normalize":
        return project_birkhoff_normalize(P, num_iters)
    raise ValueError(f"Unsupported projection method: {method}")


def binary_constraint(P, kind):
    if kind == "square":
        return P * (P - 1.0)
    if kind == "entropy":
        eps = 1e-30
        return P * torch.log(P + eps) + (1.0 - P) * torch.log(1.0 - P + eps)
    raise ValueError(f"Unsupported binary constraint: {kind}")


def binary_constraint_grad(P, kind):
    if kind == "square":
        return 2.0 * P - 1.0
    if kind == "entropy":
        eps = 1e-30
        return torch.log(P + eps) - torch.log(1.0 - P + eps)
    raise ValueError(f"Unsupported binary constraint: {kind}")


def binary_constraint_and_grad(P, kind):
    if kind == "square":
        return P * (P - 1.0), 2.0 * P - 1.0
    if kind == "entropy":
        eps = 1e-30
        log_p = torch.log(P + eps)
        log_1mp = torch.log(1.0 - P + eps)
        return P * log_p + (1.0 - P) * log_1mp, log_p - log_1mp
    raise ValueError(f"Unsupported binary constraint: {kind}")


@torch.no_grad()
def pdbo_train_step(
    P,
    Y,
    state,
    F,
    F_T,
    D,
    D_T,
    primal_lr,
    dual_lr,
    projection_iters,
    projection_method,
    binary_g,
    optimizer_name,
    track_terms,
):
    M1 = torch.matmul(F, P)
    M2 = torch.matmul(M1, D_T)

    g_val, g_grad = binary_constraint_and_grad(P, binary_g)
    if track_terms:
        term1 = torch.sum(M2 * P, dim=(1, 2)).mean()
        term2 = torch.sum(Y * g_val, dim=(1, 2)).mean()
        loss = term1 + term2
    else:
        loss = term1 = term2 = torch.zeros((), device=P.device, dtype=P.dtype)

    qap_grad = M2 + torch.matmul(torch.matmul(F_T, P), D)
    grad = (qap_grad + Y * g_grad) / P.shape[0]

    if optimizer_name == "adam":
        state["step"] += 1
        beta1 = state["beta1"]
        beta2 = state["beta2"]
        state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        step_size = primal_lr * (bias_correction2 ** 0.5) / bias_correction1
        P = P - step_size * state["exp_avg"] / (state["exp_avg_sq"].sqrt() + state["eps"])
    else:
        state["square_avg"].mul_(0.99).addcmul_(grad, grad, value=0.01)
        P = P - primal_lr * grad / (state["square_avg"].sqrt() + state["eps"])

    P = project_primal(P, projection_iters, projection_method)
    Y = Y + dual_lr * g_val
    return P.detach(), Y.detach(), loss.detach(), term1.detach(), term2.detach()


def make_optimizer_state(P, optimizer_name):
    if optimizer_name == "adam":
        return {
            "step": 0,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "exp_avg": torch.zeros_like(P),
            "exp_avg_sq": torch.zeros_like(P),
        }
    return {"eps": 1e-8, "square_avg": torch.zeros_like(P)}


def make_initial_primal(
    F_np,
    D_np,
    batch_size,
    device,
    dtype,
    init_mode,
    projection_iters,
    projection_method,
    verbose,
):
    n = F_np.shape[0]
    parts = []
    if init_mode in {"mixed", "spectral"}:
        spectral = srpd.spectral_initialization_qap(F_np, D_np, batch_size, verbose=verbose)
        if spectral.shape[0] > 0:
            parts.append(torch.tensor(spectral, device=device, dtype=dtype))

    current = sum(part.shape[0] for part in parts)
    if init_mode in {"mixed", "random"} and current < batch_size:
        num_random = batch_size - current
        random_np = np.array(srpd.latin_hypercube_matrices(n, num_random))
        parts.append(torch.tensor(random_np, device=device, dtype=dtype))

    if not parts:
        parts.append(torch.rand((batch_size, n, n), device=device, dtype=dtype))

    P = torch.cat(parts, dim=0)[:batch_size]
    return project_primal(P, projection_iters, projection_method)


def greedy_round(P, out=None):
    if P.is_cuda:
        return srpd.run_greedy_triton(P, out)
    return srpd.greedy_round_large_batch_torch(P)


def run_optimization(
    F_np,
    D_np,
    dual_init,
    batch_size,
    num_steps=100,
    primal_lr=0.02,
    dual_lr=0.02,
    optimizer_type="rmsprop",
    c_backend="on",
    eval_interval=None,
    eval_dense_until=None,
    target_obj=None,
    instance_name="",
    two_opt=False,
    two_opt_iter=None,
    two_opt_actions=None,
    two_opt_topk=None,
    two_opt_interval=1,
    projection_iters=6,
    projection_method="normalize",
    binary_g="entropy",
    init_mode="mixed",
    verbose=False,
):
    srpd._use_cuda_greedy = c_backend == "on"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    n = F_np.shape[0]
    optimizer_name = optimizer_type.lower()

    F = torch.tensor(F_np, device=device, dtype=dtype)
    F_T = F.transpose(-1, -2).contiguous()
    D = torch.tensor(D_np, device=device, dtype=dtype)
    D_T = D.transpose(-1, -2).contiguous()
    F_int = torch.tensor(F_np, device=device, dtype=torch.int64) if target_obj is not None else None
    D_int = torch.tensor(D_np, device=device, dtype=torch.int64) if target_obj is not None else None
    target_obj_int = int(target_obj) if target_obj is not None else None

    if eval_interval is None:
        eval_interval = 1
    if eval_dense_until is None:
        eval_dense_until = 0
    eval_interval = max(1, int(eval_interval))
    eval_dense_until = max(0, int(eval_dense_until))

    two_opt_iter, two_opt_actions, two_opt_topk = srpd.default_two_opt_options(
        n, batch_size, two_opt_iter, two_opt_actions, two_opt_topk
    )
    two_opt_topk = min(two_opt_topk, batch_size)
    two_opt_interval = max(1, int(two_opt_interval))

    P = make_initial_primal(
        F_np,
        D_np,
        batch_size,
        device,
        dtype,
        init_mode,
        projection_iters,
        projection_method,
        verbose,
    )
    Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    opt_state = make_optimizer_state(P, optimizer_name)

    # Warm the projection/rounding kernels without advancing the real state.
    warm_P = project_primal(
        P[: min(2, batch_size)].clone(),
        min(projection_iters, 2),
        projection_method,
    )
    _ = greedy_round(warm_P)
    if device.type == "cuda":
        torch.cuda.synchronize()

    if verbose:
        print(f"Starting PDBO [N={n}, Batch={batch_size}, Device={device}]")
        print(
            f"Projection iters={projection_iters}, binary_g={binary_g}, "
            f"projection={projection_method}, eval_interval={eval_interval}, "
            f"dense_until={eval_dense_until}"
        )
        if two_opt:
            print(
                f"2-opt enabled: iter={two_opt_iter}, actions={two_opt_actions}, "
                f"topk={two_opt_topk}, interval={two_opt_interval}"
            )

    t0 = time.time()
    incumbent_obj = float("inf")
    incumbent_X = None
    incumbents = []
    incum_time = []
    bks_time = None
    X_int_buffer = torch.empty_like(P) if device.type == "cuda" else None

    for it in range(num_steps):
        P, Y, loss, term1, term2 = pdbo_train_step(
            P,
            Y,
            opt_state,
            F,
            F_T,
            D,
            D_T,
            primal_lr,
            dual_lr,
            projection_iters,
            projection_method,
            binary_g,
            optimizer_name,
            verbose,
        )

        do_eval = it < eval_dense_until or it % eval_interval == 0 or it == num_steps - 1
        if not do_eval:
            continue

        X_int = greedy_round(P, X_int_buffer)
        perms = torch.argmax(X_int, dim=2).to(torch.int32).contiguous()
        obj_vals = srpd.compute_perm_costs(perms, F, D)

        eval_id = it if eval_interval <= 0 else it // eval_interval
        do_two_opt = two_opt and (eval_id % two_opt_interval == 0 or it == num_steps - 1)
        if do_two_opt:
            ls_count = min(two_opt_topk, perms.shape[0])
            if ls_count > 0:
                _, ls_indices = torch.topk(obj_vals, k=ls_count, largest=False)
                ls_perms = perms[ls_indices].clone().contiguous()
                ls_perms = srpd.run_two_opt_search(
                    ls_perms, F, D, max_iter=two_opt_iter, num_actions=two_opt_actions
                )
                obj_ls = srpd.compute_perm_costs(ls_perms, F, D)
                improved = obj_ls < obj_vals[ls_indices]
                if torch.any(improved):
                    improved_indices = ls_indices[improved]
                    perms[improved_indices] = ls_perms[improved]
                    obj_vals[improved_indices] = obj_ls[improved]

        min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
        should_stop = False
        if min_obj_batch < incumbent_obj:
            incumbent_obj = min_obj_batch.item()
            incumbent_X = srpd.permutations_to_assignment(
                perms[min_idx : min_idx + 1], P.dtype
            )[0]
            time_now = time.time() - t0
            incumbents.append(incumbent_obj)
            incum_time.append(time_now)

            if target_obj is not None:
                perm = perms[min_idx].long()
                exact_obj = torch.sum(F_int * D_int[perm.unsqueeze(1), perm.unsqueeze(0)])
                if exact_obj.item() == target_obj_int:
                    bks_time = time_now if bks_time is None else bks_time
                    should_stop = True

        if verbose and (it + 1) % 100 == 0:
            frac = -binary_constraint(P, binary_g).sum(dim=(1, 2)).mean().item()
            print(
                f"Iter {it+1}: Best Obj {incumbent_obj:.4f}, Loss {loss.item():.4f}, "
                f"Term1 {term1.item():.4f}, Term2 {term2.item():.4f}, Fractionality {frac:.4f}"
            )
        if should_stop:
            break

    total_time = time.time() - t0
    if incumbent_X is None:
        X_int = greedy_round(P, X_int_buffer)
        perms = torch.argmax(X_int, dim=2).to(torch.int32).contiguous()
        obj_vals = srpd.compute_perm_costs(perms, F, D)
        min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
        incumbent_obj = min_obj_batch.item()
        incumbent_X = srpd.permutations_to_assignment(perms[min_idx : min_idx + 1], P.dtype)[0]
        incumbents.append(incumbent_obj)
        incum_time.append(total_time)

    if verbose:
        print(f"Total Time: {total_time:.2f}s, FPS: {num_steps / max(total_time, 1e-9):.1f}")
    return incumbent_X, incumbent_obj, total_time, incumbents, incum_time, bks_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, default="nug12")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=["rmsprop", "adam"])
    parser.add_argument("--dual_init", type=float, default=1.0)
    parser.add_argument("--primal_lr", type=float, default=0.02)
    parser.add_argument("--dual_lr", type=float, default=0.02)
    parser.add_argument("--auto_config", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--c_backend", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_dense_until", type=int, default=None)
    parser.add_argument("--target_file", type=str, default=None)
    parser.add_argument("--target_stop", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--tf32", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--two_opt", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--two_opt_iter", type=int, default=None)
    parser.add_argument("--two_opt_actions", type=int, default=None)
    parser.add_argument("--two_opt_topk", type=int, default=None)
    parser.add_argument("--two_opt_interval", type=int, default=1)
    parser.add_argument("--projection_iters", type=int, default=6)
    parser.add_argument(
        "--projection_method",
        type=str,
        default="normalize",
        choices=["normalize", "simplex"],
    )
    parser.add_argument("--binary_g", type=str, default="entropy", choices=["entropy", "square"])
    parser.add_argument("--init", type=str, default="mixed", choices=["mixed", "spectral", "random"])
    parser.add_argument("--record_pdbo", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--warmstart", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--warmstart_iters", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    srpd.reset_random_seeds(args.seed)
    if args.tf32 == "on":
        torch.set_float32_matmul_precision("high")

    n, F_np, D_np, obj_label, _ = srpd.read_instance(args.instance)
    batch_size = args.batch_size
    num_steps = args.iters
    if args.auto_config == "on":
        if n < 200:
            batch_size = 2000
            num_steps = 1000
        elif n < 500:
            batch_size = 1000
            num_steps = 1000
        else:
            batch_size = 200
            num_steps = 1000

    target_obj = srpd.read_reference_objective(args.instance, args.target_file)
    if target_obj is None:
        target_obj = srpd.read_reference_objective(args.instance)
    if args.target_stop == "off":
        target_obj = None

    run_kwargs = dict(
        optimizer_type=args.optimizer,
        c_backend=args.c_backend,
        eval_interval=args.eval_interval,
        eval_dense_until=args.eval_dense_until,
        target_obj=target_obj,
        instance_name=args.instance.split("/")[-1],
        two_opt=args.two_opt == "on",
        two_opt_iter=args.two_opt_iter,
        two_opt_actions=args.two_opt_actions,
        two_opt_topk=args.two_opt_topk,
        two_opt_interval=args.two_opt_interval,
        projection_iters=max(1, args.projection_iters),
        projection_method=args.projection_method,
        binary_g=args.binary_g,
        init_mode=args.init,
        verbose=args.verbose,
    )

    if args.warmstart == "on":
        warmstart_iters = max(1, args.warmstart_iters)
        if args.verbose:
            print(f"Warmstart: running {warmstart_iters} iteration(s) before official timing.")
        srpd.reset_random_seeds(args.seed)
        _ = run_optimization(
            F_np,
            D_np,
            args.dual_init,
            batch_size,
            warmstart_iters,
            args.primal_lr,
            args.dual_lr,
            **run_kwargs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        srpd.reset_random_seeds(args.seed)

    start_time = time.time()
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    X_best, obj_best, solve_time_raw, incumbents, incum_time, bks_time = run_optimization(
        F_np,
        D_np,
        args.dual_init,
        batch_size,
        num_steps,
        args.primal_lr,
        args.dual_lr,
        **run_kwargs,
    )

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        solve_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        solve_time = time.time() - start_time

    row_sums = X_best.sum(axis=1)
    col_sums = X_best.sum(axis=0)
    if row_sums.min() < 0.99 or row_sums.max() > 1.01 or col_sums.min() < 0.99 or col_sums.max() > 1.01:
        print("Warning: Solution might not be a valid permutation.")
    elif args.verbose:
        print("Solution is a valid permutation matrix.")

    X_best_np = X_best.cpu().numpy()
    obj_best = np.trace(F_np @ (X_best_np @ D_np.T @ X_best_np.T))
    report_time_raw = bks_time if bks_time is not None else (incum_time[-1] if incumbents else solve_time_raw)
    report_time = report_time_raw if torch.cuda.is_available() else min(report_time_raw, solve_time)

    instance_name = args.instance.split("/")[-1]
    with open("result.txt", "a") as f:
        f.write(f"{instance_name} {report_time:.6f} {report_time_raw:.6f} {obj_best}\n")

    if args.record_pdbo == "on":
        pdbo_dir = os.path.join("results", "pdbo")
        os.makedirs(pdbo_dir, exist_ok=True)
        with open(os.path.join(pdbo_dir, f"{instance_name}.txt"), "w") as f:
            for t, val in zip(incum_time, incumbents):
                f.write(f"{t:.4f} {val:g}\n")

    if args.verbose:
        gap = (obj_best - obj_label) / obj_label if obj_label else float("nan")
        print(
            f"Finished {instance_name}: obj={obj_best:g}, label={obj_label:g}, "
            f"gap={gap:.6f}, solve_time={solve_time:.4f}s"
        )


if __name__ == "__main__":
    main()
