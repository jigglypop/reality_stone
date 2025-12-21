import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import json
import reality_stone._rust as rs_rust

class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

class GlobalManifoldLearner:
    def __init__(
        self, 
        model: nn.Module, 
        d_model: int,
        r: int = 128, 
        hyper_hidden_dim: int = 64,
        layer_emb_dim: int = 64
    ):
        self.model = model
        self.d_model = d_model
        self.r = r
        self.hyper_hidden_dim = hyper_hidden_dim
        self.layer_emb_dim = layer_emb_dim
        
        self.layers_wq = []
        self.layers_wk = []
        self.layer_indices = []
        self.layer_map = {} 
        
        self.u_global = None
        self.v_global = None
        self.hypernet = None
        self.layer_embeddings = None
        
    def collect_weights(self):
        print("Collecting weights...")
        idx = 0
        for name, module in self.model.named_modules():
            wq = None
            wk = None
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                wq = module.q_proj.weight.detach().cpu().numpy().astype(np.float32)
                wk = module.k_proj.weight.detach().cpu().numpy().astype(np.float32)
            elif hasattr(module, 'c_attn') and hasattr(module.c_attn, 'weight'):
                c_attn_w = module.c_attn.weight.detach().cpu().numpy().astype(np.float32)
                d = self.d_model
                if c_attn_w.shape == (d, 3 * d):
                    wq = c_attn_w[:, :d].T
                    wk = c_attn_w[:, d:2*d].T
                elif c_attn_w.shape == (3 * d, d):
                    wq = c_attn_w[:d, :]
                    wk = c_attn_w[d:2*d, :]
            if wq is not None and wk is not None:
                self.layers_wq.append(np.ascontiguousarray(wq))
                self.layers_wk.append(np.ascontiguousarray(wk))
                self.layer_indices.append(idx)
                self.layer_map[idx] = name
                idx += 1
                
        print(f"Collected {len(self.layers_wq)} layers.")

    def extract_global_basis(self):
        if not self.layers_wq:
            self.collect_weights()
            
        print("Extracting Global Basis (SVD)...")
        basis_dict = rs_rust.extract_global_basis(
            self.layers_wq, 
            self.layers_wk, 
            self.r
        )
        
        self.u_global = torch.from_numpy(basis_dict['u']) 
        self.v_global = self.u_global.clone() 
        
        print(f"Basis extracted. Rank: {basis_dict['rank']}")

    def train_hypernet(self, epochs=1000, batch_size=32, lr=1e-3, device=None):
        if self.u_global is None:
            self.extract_global_basis()
            
        print("Preparing Training Data (Core Tensors)...")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        u = self.u_global.to(device) 
        v = self.v_global.to(device) 
        
        targets = []
        layer_embs = []
        
        self.layer_embeddings = nn.Embedding(len(self.layers_wq), self.layer_emb_dim).to(device)
        self.hypernet = TinyMLP(self.layer_emb_dim, self.hyper_hidden_dim, self.r * self.r).to(device)
        
        optimizer = torch.optim.Adam(
            list(self.hypernet.parameters()) + list(self.layer_embeddings.parameters()), 
            lr=lr
        )
        
        with torch.no_grad():
            for i in range(len(self.layers_wq)):
                wq = torch.from_numpy(self.layers_wq[i]).to(device)
                wk = torch.from_numpy(self.layers_wk[i]).to(device)
                
                g = torch.matmul(wq.T, wk)
                
                g_sym = (g + g.T) * 0.5
                
                c = torch.matmul(torch.matmul(u.T, g_sym), u)
                
                targets.append(c.reshape(-1))
                layer_embs.append(i)
        
        targets = torch.stack(targets)
        indices = torch.tensor(layer_embs, device=device)
        
        print("Training HyperNet...")
        pbar = tqdm(range(epochs))
        loss_fn = nn.MSELoss()
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            emb = self.layer_embeddings(indices)
            pred = self.hypernet(emb)
            
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})
            
        print("HyperNet Trained.")
        
    def create_rust_hyper_metric(self):
        if self.hypernet is None:
            raise ValueError("HyperNet not trained yet.")
            
        w1 = self.hypernet.l1.weight.detach().cpu().numpy().astype(np.float32).T
        b1 = self.hypernet.l1.bias.detach().cpu().numpy().astype(np.float32)
        w2 = self.hypernet.l2.weight.detach().cpu().numpy().astype(np.float32).T
        b2 = self.hypernet.l2.bias.detach().cpu().numpy().astype(np.float32)
        
        u_np = self.u_global.detach().cpu().numpy().astype(np.float32)
        v_np = self.v_global.detach().cpu().numpy().astype(np.float32)
        
        return rs_rust.PyHyperMetric(u_np, v_np, w1, b1, w2, b2)

    def get_layer_embedding(self, idx: int):
        if self.layer_embeddings is None:
            raise ValueError("Embeddings not initialized.")
        return self.layer_embeddings(torch.tensor(idx, device=self.layer_embeddings.weight.device)).detach().cpu().numpy().astype(np.float32)

    def replace_layers(self):
        rust_hm = self.create_rust_hyper_metric()
        return SymplecticModelWrapper(self.model, self.layer_indices, rust_hm, self.layer_embeddings)

    def save_rsu_v2(self, path):
        if self.u_global is None or self.v_global is None:
            raise ValueError("Global basis not set.")
        if self.hypernet is None or self.layer_embeddings is None:
            raise ValueError("HyperNet or layer embeddings not trained.")
        path_obj = Path(path)
        header = {
            "magic": "RSULF2",
            "version": 2,
            "d_model": int(self.d_model),
            "rank": int(self.r),
            "hyper_hidden_dim": int(self.hyper_hidden_dim),
            "layer_emb_dim": int(self.layer_emb_dim),
            "num_layers": int(len(self.layers_wq) if self.layers_wq else self.layer_embeddings.num_embeddings),
            "model_type": type(self.model).__name__,
        }
        u_np = self.u_global.detach().cpu().numpy().astype(np.float32)
        v_np = self.v_global.detach().cpu().numpy().astype(np.float32)
        w1_np = self.hypernet.l1.weight.detach().cpu().numpy().astype(np.float32)
        b1_np = self.hypernet.l1.bias.detach().cpu().numpy().astype(np.float32)
        w2_np = self.hypernet.l2.weight.detach().cpu().numpy().astype(np.float32)
        b2_np = self.hypernet.l2.bias.detach().cpu().numpy().astype(np.float32)
        emb_np = self.layer_embeddings.weight.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(
            str(path_obj),
            header=json.dumps(header),
            u=u_np,
            v=v_np,
            w1=w1_np,
            b1=b1_np,
            w2=w2_np,
            b2=b2_np,
            layer_embeddings=emb_np,
        )

    @classmethod
    def from_rsu_v2(
        cls,
        model: nn.Module,
        path,
    ):
        path_obj = Path(path)
        data = np.load(str(path_obj), allow_pickle=False)
        header_raw = data["header"].item() if isinstance(data["header"], np.ndarray) else data["header"]
        header = json.loads(header_raw)
        d_model = int(header.get("d_model", 0))
        rank = int(header.get("rank", 0))
        hyper_hidden_dim = int(header.get("hyper_hidden_dim", 0))
        layer_emb_dim = int(header.get("layer_emb_dim", 0))
        num_layers = int(header.get("num_layers", 0))
        learner = cls(
            model=model,
            d_model=d_model,
            r=rank,
            hyper_hidden_dim=hyper_hidden_dim,
            layer_emb_dim=layer_emb_dim,
        )
        learner.u_global = torch.from_numpy(data["u"])
        learner.v_global = torch.from_numpy(data["v"])
        learner.layers_wq = []
        learner.layers_wk = []
        learner.layer_indices = list(range(num_layers))
        learner.layer_map = {i: str(i) for i in range(num_layers)}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hypernet = TinyMLP(layer_emb_dim, hyper_hidden_dim, rank * rank)
        hypernet.l1.weight.data.copy_(torch.from_numpy(data["w1"]).to(hypernet.l1.weight.dtype))
        hypernet.l1.bias.data.copy_(torch.from_numpy(data["b1"]).to(hypernet.l1.bias.dtype))
        hypernet.l2.weight.data.copy_(torch.from_numpy(data["w2"]).to(hypernet.l2.weight.dtype))
        hypernet.l2.bias.data.copy_(torch.from_numpy(data["b2"]).to(hypernet.l2.bias.dtype))
        learner.hypernet = hypernet.to(device)
        emb_weight = torch.from_numpy(data["layer_embeddings"])
        num_embeddings, emb_dim = emb_weight.shape
        embedding = nn.Embedding(num_embeddings, emb_dim)
        embedding.weight.data.copy_(emb_weight)
        learner.layer_embeddings = embedding.to(device)
        return learner

class SymplecticModelWrapper(nn.Module):
    def __init__(self, original_model, layer_indices, rust_hyper_metric, layer_embeddings):
        super().__init__()
        self.original_model = original_model
        self.layer_indices = set(layer_indices)
        self.hyper_metric = rust_hyper_metric
        self.layer_embeddings = layer_embeddings
        self.dt = 0.01
        
        self.symplectic_layers = {}
        device = layer_embeddings.weight.device
        
        for idx in layer_indices:
            emb = layer_embeddings(torch.tensor(idx, device=device)).detach().cpu().numpy().astype(np.float32)
            
            self.symplectic_layers[idx] = rs_rust.PySymplecticLayer(
                layer_idx=idx,
                layer_emb=emb,
                hyper_metric=rust_hyper_metric,
                dt=self.dt
            )

    def _get_layers(self):
        if hasattr(self.original_model, 'layers'):
            return list(self.original_model.layers)
        if hasattr(self.original_model, 'transformer') and hasattr(self.original_model.transformer, 'h'):
            return list(self.original_model.transformer.h)
        if hasattr(self.original_model, 'model') and hasattr(self.original_model.model, 'layers'):
            return list(self.original_model.model.layers)
        raise AttributeError("Could not find transformer layers")
            
    def forward(self, x):
        q = x
        p = torch.zeros_like(q)
        layers = self._get_layers()
        
        for i, layer in enumerate(layers):
            if i in self.symplectic_layers:
                out = layer(q)
                base_out = out[0] if isinstance(out, (tuple, list)) else out
                kick = base_out - q
                q_np = q.detach().cpu().numpy().astype(np.float32)
                p_np = p.detach().cpu().numpy().astype(np.float32)
                kick_np = kick.detach().cpu().numpy().astype(np.float32)
                
                q_out_np, p_out_np = self.symplectic_layers[i].step(q_np, p_np, kick_np)
                
                q = torch.from_numpy(q_out_np).to(q.device)
                p = torch.from_numpy(p_out_np).to(p.device)
            else:
                out = layer(q)
                q = out[0] if isinstance(out, (tuple, list)) else out
                
        return q
