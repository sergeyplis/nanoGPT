import torch
import torch.func as func


class FixedPointIteration(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, z_init, max_iter=100, tol=1e-5, num_heads=1):
        z = z_init
        B, N, C = z.shape
        H = num_heads
        D = C // H

        with torch.no_grad():
            allconverged_indicator = False
            for _ in range(max_iter):
                z_next = f(x, z)
                diff = (z_next - z).view(B, N, H, D)
                diff_norm = torch.norm(diff, dim=-1)  # [B, N, H]
                diff_norm = diff_norm.max(dim=1).values  # [B, H]
                just_converged = diff_norm < tol

                if _ == 0:
                    converged = just_converged
                else:
                    converged |= just_converged

                update_mask = ~converged.view(B, 1, H, 1).expand(B, N, H, D)
                update_mask = update_mask.reshape(B, N, C)
                z = torch.where(update_mask, z_next, z)

                if torch.all(converged):
                    # print(f'forward: convergef in {_}')
                    allconverged_indicator = True
                    break
        # if not allconverged_indicator:
        #    print("forward has not converged")
        ctx.save_for_backward(x, z)
        ctx.f = f
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.num_heads = num_heads
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, z_star = ctx.saved_tensors
        f = ctx.f
        max_iter = ctx.max_iter
        tol = ctx.tol
        num_heads = ctx.num_heads

        B, N, C = z_star.shape
        H = num_heads
        D = C // H

        def f_z(z):
            return f(x, z)

        _, vjp_fn = func.vjp(f_z, z_star)

        allconverged_indicator = False
        adjoint = grad_output
        for i in range(max_iter):
            JTv = vjp_fn(adjoint)[0]
            adjoint_next = grad_output + JTv

            diff = (adjoint_next - adjoint).view(B, N, H, D)
            diff_norm = torch.norm(diff, dim=-1)  # [B, N, H]
            diff_norm = diff_norm.max(dim=1).values  # [B, H]
            just_converged = diff_norm < tol

            if i == 0:
                converged = just_converged
            else:
                converged |= just_converged

            update_mask = ~converged.view(B, 1, H, 1).expand(B, N, H, D)
            update_mask = update_mask.reshape(B, N, C)
            adjoint = torch.where(update_mask, adjoint_next, adjoint)

            if torch.all(converged):
                # print(f"{i}")
                allconverged_indicator = True
                break

        #        if not allconverged_indicator:
        #            print('Backward has not converged')

        def f_x(x):
            return f(x, z_star)

        _, vjp_x_fn = func.vjp(f_x, x)
        grad_x = vjp_x_fn(adjoint)[0]

        return None, grad_x, None, None, None, None
