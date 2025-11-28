import torch
import torch.nn as nn
import reality_stone as rs


class RiemannianDiffusionStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, flow, diffusion_engine, alpha, dt):
        h = h.contiguous()
        flow = flow.contiguous()

        h_next = torch.empty_like(h)
        batch_size, dim = h.shape

        if h.is_cuda and getattr(rs, "_has_cuda", False):
            diffusion_engine.step_cuda(
                h.data_ptr(),
                flow.data_ptr(),
                h_next.data_ptr(),
                batch_size,
                dim,
            )
        else:
            h_np = h.detach().cpu().numpy().astype("float32")
            flow_np = flow.detach().cpu().numpy().astype("float32")
            h_next_np = diffusion_engine.step_cpu(h_np, flow_np)
            h_next = torch.from_numpy(h_next_np).to(h.device)

        ctx.alpha = float(alpha)
        ctx.dt = float(dt)
        return h_next

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        dt = ctx.dt
        a = 1.0 - (1.0 - alpha) * dt
        b = (1.0 - alpha) * dt

        grad_h = grad_output * a
        grad_flow = grad_output * b

        return grad_h, grad_flow, None, None, None


class RiemannianDiffusionModule(nn.Module):
    def __init__(self, dim, alpha=0.9, dt=0.1, num_steps=5):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.dt = dt
        self.num_steps = num_steps
        
        if rs.PyRiemannianDiffusion is not None:
            self.engine = rs.PyRiemannianDiffusion(dim, alpha, dt)
        else:
            self.engine = None
        
        self.flow_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h):
        if self.engine is None:
            return h
            
        for _ in range(self.num_steps):
            flow = self.flow_net(h)
            h = RiemannianDiffusionStep.apply(h, flow, self.engine, self.alpha, self.dt)
        return h

