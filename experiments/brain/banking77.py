import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.manifold import MDS
import pandas as pd
import reality_stone as rs

torch.manual_seed(42)
np.random.seed(42)

haxby_dataset = datasets.fetch_haxby()
func_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask_vt[0]

masker = NiftiMasker(mask_img=mask_filename, standardize=True, detrend=True)
fmri_masked = masker.fit_transform(func_filename)

behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
conditions = behavioral["labels"]

target_conditions = ["face", "house", "cat", "shoe", "bottle"]
condition_mask = conditions.isin(target_conditions)
fmri_data = fmri_masked[condition_mask]
labels = conditions[condition_mask].values

n_samples = 150
fmri_data = fmri_data[:n_samples]
labels = labels[:n_samples]

unique_labels, label_indices = np.unique(labels, return_inverse=True)
num_cond = len(unique_labels)

# NaN 처리 추가
fmri_data = np.nan_to_num(fmri_data)

brain_cond = []
for k in range(num_cond):
    brain_cond.append(fmri_data[label_indices == k].mean(axis=0))
brain_cond = np.stack(brain_cond, axis=0)

def rdm_cosine(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x)
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x = x / denom
    sim = x @ x.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim

brain_rdm_cond = rdm_cosine(brain_cond)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = 32
prototypes_tan = nn.Parameter(torch.randn(num_cond, input_dim, device=device))
optimizer = torch.optim.Adam([prototypes_tan], lr=1e-2)

brain_rdm_t = torch.tensor(brain_rdm_cond, dtype=torch.float32, device=device)
triu_idx = torch.triu_indices(num_cond, num_cond, offset=1)
c = 1.0

def hyperbolic_rdm(proto_tan):
    proto = rs.layers.poincare.exp_map_zero(proto_tan, c=c)  # R^d -> Poincaré
    a = proto.unsqueeze(0).expand(num_cond, -1, -1)
    b = proto.unsqueeze(1).expand(-1, num_cond, -1)
    dist = rs.poincare_distance(
        a.reshape(-1, input_dim),
        b.reshape(-1, input_dim),
        c
    )
    return dist.view(num_cond, num_cond)

class RiemannianDynamics(nn.Module):
    def __init__(self, curvature: float, steps: int = 20, lr: float = 0.1, beta: float = 0.9, tau: float = 1.0):
        super().__init__()
        self.c = curvature
        self.steps = steps
        self.lr = lr
        self.beta = beta
        self.tau = tau

    def forward(self, z0: torch.Tensor, mu_hyp: torch.Tensor) -> torch.Tensor:
        # z0: (N, d) tangent at origin, requires_grad False OK
        # mu_hyp: (N, d) Poincaré points (targets per sample)
        z = z0.detach().clone().requires_grad_(True)
        v = torch.zeros_like(z)
        for _ in range(self.steps):
            q = rs.layers.poincare.exp_map_zero(z, c=self.c)
            d = rs.poincare_distance(q, mu_hyp, self.c)  # (N,)
            phi = self.tau * (d * d).mean()
            g = torch.autograd.grad(phi, z, only_inputs=True)[0]
            v = self.beta * v - self.lr * g
            z = (z + v).detach().requires_grad_(True)
        return rs.layers.poincare.exp_map_zero(z, c=self.c).detach()

steps = 2000
for step in range(steps):
    optimizer.zero_grad()
    model_rdm = hyperbolic_rdm(prototypes_tan)
    b_vec = brain_rdm_t[triu_idx[0], triu_idx[1]]
    m_vec = model_rdm[triu_idx[0], triu_idx[1]]

    valid = torch.isfinite(b_vec) & torch.isfinite(m_vec)
    if valid.sum() == 0:
        continue

    b = b_vec[valid]
    m = m_vec[valid]

    b = (b - b.mean()) / (b.std() + 1e-8)
    m = (m - m.mean()) / (m.std() + 1e-8)
    corr = (b * m).mean()

    # Anti-collapse regularization
    var_term = m.std()  # encourage dispersion of model distances
    norms = prototypes_tan.norm(dim=1)
    reg_small = F.relu(0.2 - norms).mean()
    reg_large = F.relu(norms - 2.0).mean()

    loss = -corr - 0.05 * var_term + 0.01 * (reg_small + reg_large)
    loss.backward()
    optimizer.step()
    if (step + 1) % 200 == 0:
        print(step + 1, "cond-level hyperbolic RSA", corr.item())

with torch.no_grad():
    proto_tan_final = prototypes_tan.detach().to(device)
    proto_final = rs.layers.poincare.exp_map_zero(proto_tan_final, c=c).cpu().numpy()

# Sample states via Riemannian Dynamics
idx = torch.tensor(label_indices, dtype=torch.long, device=device)
mu_hyp = rs.layers.poincare.exp_map_zero(prototypes_tan[idx], c=c)
z0 = torch.randn(len(labels), input_dim, device=device) * 0.3
solver = RiemannianDynamics(curvature=c, steps=20, lr=0.1, beta=0.9, tau=1.0).to(device)
model_states = solver(z0, mu_hyp)
model_data = model_states.cpu().numpy()

# MDS 입력 전 NaN 처리
fmri_data = np.nan_to_num(fmri_data)
model_data = np.nan_to_num(model_data)

brain_rdm_full = rdm_cosine(fmri_data)

# Poincaré pairwise RDM for model
def rdm_poincare_numpy(x_np: np.ndarray) -> np.ndarray:
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    n = x.shape[0]
    a = x.unsqueeze(0).expand(n, -1, -1)
    b = x.unsqueeze(1).expand(-1, n, -1)
    d = rs.poincare_distance(a.reshape(-1, input_dim), b.reshape(-1, input_dim), c).view(n, n)
    return d.cpu().numpy()

model_rdm_full = rdm_poincare_numpy(model_data)

tri_full = np.triu_indices(n_samples, k=1)
u = brain_rdm_full[tri_full]
v = model_rdm_full[tri_full]
mask = np.isfinite(u) & np.isfinite(v)
u, v = u[mask], v[mask]
if u.size == 0 or v.size == 0 or u.std() < 1e-12 or v.std() < 1e-12:
    rsa_score = 0.0
else:
    u = (u - u.mean()) / (u.std() + 1e-8)
    v = (v - v.mean()) / (v.std() + 1e-8)
    rsa_score = float((u * v).mean())
print("sample-level RSA (brain: cosine, model: Poincaré):", rsa_score)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.regplot(
    x=brain_rdm_full[tri_full],
    y=model_rdm_full[tri_full],
    ax=axes[0],
    scatter_kws={"alpha": 0.05, "color": "black", "s": 2},
    line_kws={"color": "red"},
)
axes[0].set_title(f"RSA Score: {rsa_score:.4f}")
axes[0].set_xlabel("Real Brain Dissimilarity")
axes[0].set_ylabel("Hyperbolic Model Dissimilarity")

mds = MDS(n_components=2, random_state=42, dissimilarity="precomputed", normalized_stress="auto")
brain_2d = mds.fit_transform(np.nan_to_num(brain_rdm_full))
sns.scatterplot(
    x=brain_2d[:, 0],
    y=brain_2d[:, 1],
    hue=labels,
    ax=axes[1],
    palette="Set2",
    s=80,
    alpha=0.8,
)
axes[1].set_title("Real Brain (fMRI)")
axes[1].legend(loc="upper right", bbox_to_anchor=(1.2, 1), fontsize="small")

model_2d = mds.fit_transform(np.nan_to_num(model_rdm_full))
sns.scatterplot(
    x=model_2d[:, 0],
    y=model_2d[:, 1],
    hue=labels,
    ax=axes[2],
    palette="Set2",
    s=80,
    alpha=0.8,
)
axes[2].set_title("Hyperbolic Model (RS)")
axes[2].get_legend().remove()

plt.tight_layout()
plt.savefig("haxby_rsa_result.png")
print("Saved result to haxby_rsa_result.png")
