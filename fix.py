import torch
import torch.func as func


class FixedPointIteration(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, module, x, z_init, max_iter=100, tol=1e-5, num_heads=1, check_interval=1
    ):
        z = z_init
        B, N, C = z.shape
        H = num_heads
        D = C // H

        with torch.no_grad():
            allconverged_indicator = False
            for _ in range(max_iter):
                z_next = module(z, x)
                diff = (z_next - z).view(B, N, H, D)
                diff_norm = torch.norm(diff, dim=-1)  # [B, N, H]
                # diff_norm = diff_norm.max(dim=1).values  # [B, H]
                just_converged = diff_norm < tol

                if _ == 0:
                    converged = just_converged
                else:
                    converged |= just_converged

                update_mask = ~converged.view(B, N, H, 1).expand(B, N, H, D)
                update_mask = update_mask.reshape(B, N, C)
                z = torch.where(update_mask, z_next, z)

                if _ % check_interval == 0:
                    if torch.all(converged):
                        allconverged_indicator = True
                        break
        if not allconverged_indicator:
            print("forward has not converged")
        ctx.save_for_backward(x, z)
        ctx.module = module
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.num_heads = num_heads
        ctx.check_interval = check_interval
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, z_star = ctx.saved_tensors
        module = ctx.module
        max_iter = ctx.max_iter
        tol = ctx.tol
        num_heads = ctx.num_heads
        check_interval = ctx.check_interval

        B, N, C = z_star.shape
        H = num_heads
        D = C // H

        def f_z(z):
            return module(z, x)

        _, vjp_fn = func.vjp(f_z, z_star)

        allconverged_indicator = False
        adjoint = grad_output
        for i in range(max_iter):
            JTv = vjp_fn(adjoint)[0]
            adjoint_next = grad_output + JTv

            diff = (adjoint_next - adjoint).view(B, N, H, D)
            diff_norm = torch.norm(diff, dim=-1)  # [B, N, H]
            # diff_norm = diff_norm.max(dim=1).values  # [B, H]
            just_converged = diff_norm < tol

            if i == 0:
                converged = just_converged
            else:
                converged |= just_converged

            update_mask = ~converged.view(B, N, H, 1).expand(B, N, H, D)
            update_mask = update_mask.reshape(B, N, C)
            adjoint = torch.where(update_mask, adjoint_next, adjoint)

            if i % check_interval == 0:
                if torch.all(converged):
                    allconverged_indicator = True
                    break

        #        if not allconverged_indicator:
        #            print('Backward has not converged')

        # ────────────────────────────────
        # 2) one‑shot true backward through module parameters
        # ────────────────────────────────
        with torch.enable_grad():
            # make z_star a leaf that *does* require grad
            z_star_req = z_star.detach().requires_grad_(True)
            # re‑compute the step under grad mode
            z_next = module(z_star_req, x)
            # and now propagate adjoint into all params
            torch.autograd.grad(
                outputs=z_next,
                inputs=tuple(module.parameters()),
                grad_outputs=adjoint,
                only_inputs=True,
            )
        # ────────────────────────────────

        def f_x(x_local):
            return module(z_star, x_local)

        _, vjp_x_fn = func.vjp(f_x, x)
        grad_x = vjp_x_fn(adjoint)[0]

        return None, grad_x, None, None, None, None
