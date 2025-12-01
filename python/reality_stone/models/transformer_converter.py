import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from reality_stone._rust import (
        PyRSULFLayer,
        verify_metric_consistency,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from reality_stone.layers.rsulf_cuda import RSULFLayerCUDA, RSULFWrapperCUDA
import torch
from tqdm import tqdm


@dataclass
class ConversionStats:
    total_layers: int = 0
    converted: int = 0
    failed: List[int] = field(default_factory=list)
    layer_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    original_params: int = 0
    compressed_params: int = 0


class RSULFTransformerConverter:
    def __init__(
        self,
        d_model: int = 4096,
        r: int = 1024,
        eta: float = 0.01,
        alpha: float = 0.02,
        beta: float = 0.01,
        gamma: float = 0.99,
        seq_len: int = 128,
        window: int = 8,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 4,
        verbose: bool = False,
    ):
        if not HAS_RUST:
            raise RuntimeError("reality_stone._rust not available")
        
        self.d_model = d_model
        self.r = r
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seq_len = seq_len
        self.window = window
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        self.stats = ConversionStats()

    def extract_weights(self, layer) -> Dict[str, np.ndarray]:
        wq = layer.self_attn.q_proj.weight.detach().cpu().numpy().astype(np.float32)
        wk = layer.self_attn.k_proj.weight.detach().cpu().numpy().astype(np.float32)
        
        if wk.shape[0] < wq.shape[0]:
            repeat = wq.shape[0] // wk.shape[0]
            wk = np.tile(wk, (repeat, 1))
        
        if hasattr(layer.mlp, 'gate_proj'):
            w1 = layer.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
            w2 = layer.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
        else:
            w1 = layer.mlp.fc1.weight.detach().cpu().numpy().astype(np.float32)
            w2 = layer.mlp.fc2.weight.detach().cpu().numpy().astype(np.float32)
        
        return {"WQ": wq, "WK": wk, "W1": w1, "W2": w2}

    def verify_weights(self, weights: Dict[str, np.ndarray], idx: int) -> Tuple[bool, Dict]:
        result = {"valid": True, "issues": []}
        for name, w in weights.items():
            if np.isnan(w).any():
                result["valid"] = False
                result["issues"].append(f"{name} NaN")
            if np.isinf(w).any():
                result["valid"] = False
                result["issues"].append(f"{name} Inf")
        return result["valid"], result

    def convert_layer(self, layer, idx: int) -> Tuple[Optional[RSULFLayerCUDA], Dict[str, Any]]:
        layer_stat = {"idx": idx, "success": False}
        
        try:
            weights = self.extract_weights(layer)
            valid, check = self.verify_weights(weights, idx)
            if not valid:
                layer_stat["error"] = f"weight_verify: {check['issues']}"
                return None, layer_stat

            d_out, d_model = weights["WQ"].shape
            base_r = min(self.r, d_model)
            candidate_rs = []
            for k in [
                base_r,
                base_r * 2,
                max(base_r, d_model // 8),
                max(base_r, d_model // 4),
                max(base_r, d_model // 2),
                d_model,
            ]:
                k_clamped = int(max(1, min(d_model, k)))
                if k_clamped not in candidate_rs:
                    candidate_rs.append(k_clamped)
            candidate_rs = sorted(candidate_rs)

            best_r = None
            best_consistency = None
            for r_try in candidate_rs:
                c = verify_metric_consistency(
                    weights["WQ"], weights["WK"], int(r_try)
                )
                if c["is_valid"]:
                    best_r = int(r_try)
                    best_consistency = c
                    break
                if best_consistency is None or float(c["fold_accuracy"]) > float(best_consistency["fold_accuracy"]):
                    best_consistency = c

            if best_r is None:
                layer_stat["fold_accuracy"] = float(best_consistency["fold_accuracy"])
                layer_stat["symmetry_error"] = float(best_consistency["symmetry_error"])
                layer_stat["error"] = f"metric_invalid: fold_acc={best_consistency['fold_accuracy']:.4f}"
                return None, layer_stat

            layer_stat["r"] = best_r
            layer_stat["fold_accuracy"] = float(best_consistency["fold_accuracy"])
            layer_stat["symmetry_error"] = float(best_consistency["symmetry_error"])

            rsulf = RSULFLayerCUDA(
                wq=weights["WQ"],
                wk=weights["WK"],
                w1=weights["W1"],
                w2=weights["W2"],
                d_model=self.d_model,
                r=best_r,
                eta=self.eta,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                seq_len=self.seq_len,
                window=self.window,
            )
            
            compressed, original, ratio = rsulf.param_count()
            layer_stat["compressed"] = compressed
            layer_stat["original"] = original
            layer_stat["ratio"] = float(ratio)
            layer_stat["curvature"] = float(rsulf.curvature)
            layer_stat["eta"] = float(rsulf.eta)
            layer_stat["alpha"] = float(rsulf.alpha)
            
            x_test = torch.randn(4, self.d_model)
            out, _ = rsulf(x_test)
            if torch.isnan(out).any() or torch.isinf(out).any():
                layer_stat["error"] = "forward_nan"
                return None, layer_stat
            
            layer_stat["success"] = True
            return rsulf, layer_stat
            
        except Exception as e:
            layer_stat["error"] = str(e)
            return None, layer_stat

    def convert_model(self, model) -> Tuple[List[RSULFLayerCUDA], ConversionStats]:
        transformer_layers = model.model.layers
        self.stats.total_layers = len(transformer_layers)
        
        if self.d_model == 4096:
            self.d_model = model.config.hidden_size
        
        layers = []
        
        pbar = tqdm(transformer_layers, desc="Converting", unit="layer")
        for idx, layer in enumerate(pbar):
            pbar.set_postfix({"ok": self.stats.converted, "fail": len(self.stats.failed)})
            
            rsulf, stat = self.convert_layer(layer, idx)
            self.stats.layer_stats[idx] = stat

            if self.verbose:
                if stat.get("success"):
                    r_used = stat.get("r", self.r)
                    ratio = stat.get("ratio", 0.0)
                    fold_acc = stat.get("fold_accuracy", 0.0)
                    sym_err = stat.get("symmetry_error", 0.0)
                    print(
                        f"[RSULF] layer {idx:02d}: ok "
                        f"r={r_used}, ratio={ratio:.1f}x, "
                        f"fold_acc={fold_acc:.3f}, sym_err={sym_err:.3e}"
                    )
                else:
                    print(f"[RSULF] layer {idx:02d}: fail {stat.get('error', 'unknown')}")
            
            if rsulf is not None:
                layers.append(rsulf)
                self.stats.converted += 1
                self.stats.original_params += stat.get("original", 0)
                self.stats.compressed_params += stat.get("compressed", 0)
            else:
                self.stats.failed.append(idx)
                self.stats.errors.append({"layer": idx, "error": stat.get("error", "unknown")})
            
            if self.checkpoint_dir and (idx + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(layers, idx + 1)
        
        return layers, self.stats

    def _save_checkpoint(self, layers: List[RSULFLayerCUDA], count: int):
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{count}.json")
        data = {
            "count": count,
            "stats": {
                "converted": self.stats.converted,
                "failed": self.stats.failed,
                "original_params": self.stats.original_params,
                "compressed_params": self.stats.compressed_params,
            },
            "layers": []
        }
        for layer in layers:
            comp = layer._layer.export_components()
            layer_data = {}
            for k, v in comp.items():
                if isinstance(v, np.ndarray):
                    layer_data[k] = v.tolist()
                else:
                    layer_data[k] = v
            data["layers"].append(layer_data)
        
        with open(path, "w") as f:
            json.dump(data, f)

    def analyze_errors(self) -> Dict[str, Any]:
        analysis = {"total": len(self.stats.errors), "by_type": {}}
        for err in self.stats.errors:
            msg = err.get("error", "unknown")
            key = msg.split(":")[0] if ":" in msg else msg
            if key not in analysis["by_type"]:
                analysis["by_type"][key] = []
            analysis["by_type"][key].append(err["layer"])
        return analysis

    def verify_conversion(self, wq: np.ndarray, wk: np.ndarray) -> Dict:
        return verify_metric_consistency(
            wq.astype(np.float32),
            wk.astype(np.float32),
            self.r
        )


class RSULFStack(torch.nn.Module):
    def __init__(self, layers: List[RSULFLayerCUDA]):
        super().__init__()
        self.wrappers = torch.nn.ModuleList([
            RSULFWrapperCUDA(layer) for layer in layers
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for wrapper in self.wrappers:
            x = wrapper(x)
        return x

    def reset_memory(self):
        for wrapper in self.wrappers:
            wrapper.v_mem = None

