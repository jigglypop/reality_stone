import torch, reality_stone as rs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(4, 128, device=device, requires_grad=True)
u = torch.randn(4, 128, device=device, requires_grad=True)
t = torch.tensor(0.7, device=device, requires_grad=True)
# c는 상수로 쓰셔도 되지만, backward에서 참조된다면 tensor로 바꿔보세요
c = torch.tensor(1e-3, device=device, requires_grad=False)

y = rs.poincare_ball_layer(x, u, c, t)
print("y.grad_fn:", y.grad_fn)
loss = y.sum()
loss.backward()
print("x.grad norm:", x.grad.norm().item(), "u.grad norm:", u.grad.norm().item(), "t.grad:", t.grad)
