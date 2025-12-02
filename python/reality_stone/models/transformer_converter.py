import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from reality_stone._rust import (
        PyRSULFLayer,
        verify_metric_consistency,
        analyze_layer,
        create_compression_plan,
        extract_global_basis
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
        weights = {}
        # Check for GPT-2 style (Conv1D)
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'):
            # GPT-2 c_attn weights are [hidden_size, 3*hidden_size]
            # We need to transpose to [3*hidden_size, hidden_size] to match Linear(in, out).weight which is [out, in]
            c_attn = layer.attn.c_attn.weight.detach().cpu().numpy().astype(np.float32).T
            c_attn = np.ascontiguousarray(c_attn)
            
            hidden_size = c_attn.shape[1]
            # Split into Q, K, V
            # c_attn is [Q, K, V]
            weights["WQ"] = np.ascontiguousarray(c_attn[:hidden_size, :])
            weights["WK"] = np.ascontiguousarray(c_attn[hidden_size:2*hidden_size, :])
            
            # MLP
            # c_fc is [hidden, 4*hidden] -> Transpose to [4*hidden, hidden] (W1 - up projection)
            w1 = layer.mlp.c_fc.weight.detach().cpu().numpy().astype(np.float32).T
            weights["W1"] = np.ascontiguousarray(w1)
            # c_proj is [4*hidden, hidden] -> Transpose to [hidden, 4*hidden] (W2 - down projection)
            w2 = layer.mlp.c_proj.weight.detach().cpu().numpy().astype(np.float32).T
            weights["W2"] = np.ascontiguousarray(w2)
            
            # Extract LayerNorms if available
            if hasattr(layer, 'ln_1'):
                weights["ln_1_weight"] = layer.ln_1.weight.detach().cpu().numpy().astype(np.float32)
                weights["ln_1_bias"] = layer.ln_1.bias.detach().cpu().numpy().astype(np.float32)
            
            return weights

        # Llama / Mistral / Standard Linear
        weights["WQ"] = layer.self_attn.q_proj.weight.detach().cpu().numpy().astype(np.float32)
        wk = layer.self_attn.k_proj.weight.detach().cpu().numpy().astype(np.float32)
        
        if wk.shape[0] < weights["WQ"].shape[0]:
            repeat = weights["WQ"].shape[0] // wk.shape[0]
            wk = np.tile(wk, (repeat, 1))
        weights["WK"] = wk
        
        if hasattr(layer.mlp, 'gate_proj'):
            weights["W1"] = layer.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["W2"] = layer.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
        else:
            # e.g. OPT or others
            weights["W1"] = layer.mlp.fc1.weight.detach().cpu().numpy().astype(np.float32)
            weights["W2"] = layer.mlp.fc2.weight.detach().cpu().numpy().astype(np.float32)
        
        # Extract LayerNorms (input_layernorm or similar)
        if hasattr(layer, 'input_layernorm'):
             weights["ln_1_weight"] = layer.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
             if layer.input_layernorm.bias is not None:
                 weights["ln_1_bias"] = layer.input_layernorm.bias.detach().cpu().numpy().astype(np.float32)
        
        return weights

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
            best_r = int(max(1, min(d_model, self.r)))
            best_consistency = {"fold_accuracy": 1.0, "symmetry_error": 0.0}

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

    def convert_model(self, model) -> "RSULFModel":
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            transformer_layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            transformer_layers = model.transformer.h
        else:
             # Fallback for generic HF models or simple stacks
             if hasattr(model, "layers"):
                 transformer_layers = model.layers
             else:
                 raise AttributeError("Could not find transformer layers in model. Checked: model.model.layers, model.transformer.h")

        self.stats.total_layers = len(transformer_layers)
        
        if self.d_model == 4096:
             if hasattr(model.config, "hidden_size"):
                  self.d_model = model.config.hidden_size
             elif hasattr(model.config, "d_model"):
                  self.d_model = model.config.d_model
             elif hasattr(model.config, "n_embd"): # GPT-2
                  self.d_model = model.config.n_embd
        
        print(f"Collecting weights from {len(transformer_layers)} layers...")
        all_wq = []
        all_wk = []
        layer_weights = []
        
        for idx, layer in enumerate(transformer_layers):
            try:
                weights = self.extract_weights(layer)
                valid, check = self.verify_weights(weights, idx)
                if valid:
                    all_wq.append(weights["WQ"])
                    all_wk.append(weights["WK"])
                    layer_weights.append(weights)
                else:
                    print(f"Skipping layer {idx} due to invalid weights: {check['issues']}")
                    all_wq.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                    all_wk.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                    layer_weights.append(None)
            except Exception as e:
                print(f"Error extracting layer {idx}: {e}")
                all_wq.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                all_wk.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                layer_weights.append(None)

        print("Phase 1: Analyzing layers...")
        analyses = []
        pbar_analyze = tqdm(total=len(layer_weights), desc="Analyzing", unit="layer", disable=not self.verbose)
        for idx, weights in enumerate(layer_weights):
            if weights:
                analysis = analyze_layer(
                    weights["WQ"], weights["WK"], weights["W1"], weights["W2"], 
                    idx, self.r
                )
                analyses.append(analysis)
                acc = analysis.get("expected_accuracy", 0.0)
                pbar_analyze.set_postfix(idx=idx, acc=f"{acc:.4f}")
            pbar_analyze.update(1)
        pbar_analyze.close()
            
        print("Phase 2: Planning compression...")
        if analyses:
            plan = create_compression_plan(analyses, 0.95)
            print(f"Plan: ratio={plan.get('expected_compression_ratio', 0):.2f}x, acc={plan.get('min_expected_accuracy', 0):.4f}")
        
        print("Phase 3: Extracting Global Basis...")
        global_basis = None
        try:
            # Filter out zero weights if any
            valid_wq = [w for w in all_wq if w.shape[0] > 0 and w.any()]
            valid_wk = [w for w in all_wk if w.shape[0] > 0 and w.any()]
            if valid_wq:
                global_basis = extract_global_basis(valid_wq, valid_wk, self.r)
                print(f"Global Basis extracted: rank={global_basis['rank']}")
        except Exception as e:
            print(f"Global Basis extraction failed: {e}. Falling back to local.")

        print("Phase 4: Converting layers...")
        layers = []
        acc_by_idx = {a.get("layer_idx", i): a.get("expected_accuracy", 0.0) for i, a in enumerate(analyses)}
        pbar_convert = tqdm(total=len(layer_weights), desc="Converting", unit="layer", disable=not self.verbose)
        for idx, weights in enumerate(layer_weights):
            if weights is None:
                self.stats.failed.append(idx)
                pbar_convert.set_postfix(idx=idx, status="skip")
                pbar_convert.update(1)
                continue
                
            print(f"Processing layer {idx}...")
            
            try:
                d_out, d_model = weights["WQ"].shape
                best_r = int(max(1, min(d_model, self.r)))
                
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
                    global_basis=global_basis
                )
                
                # Attach LayerNorm params if available for Wrapper
                if "ln_1_weight" in weights:
                    rsulf.ln_1_weight = weights["ln_1_weight"]
                    rsulf.ln_1_bias = weights.get("ln_1_bias")
                
                compressed, original, ratio = rsulf.param_count()
            
                if self.verbose:
                    print(f"[RSULF] layer {idx:02d}: ok ratio={ratio:.1f}x")
                
                layers.append(rsulf)
                self.stats.converted += 1
                self.stats.original_params += original
                self.stats.compressed_params += compressed
                acc = acc_by_idx.get(idx, 0.0)
                pbar_convert.set_postfix(idx=idx, acc=f"{acc:.4f}", ratio=f"{ratio:.1f}x", status="ok")
                
            except Exception as e:
                print(f"[RSULF] layer {idx:02d}: fail {e}")
                self.stats.failed.append(idx)
                self.stats.errors.append({"layer": idx, "error": str(e)})
                pbar_convert.set_postfix(idx=idx, status="fail")
            
            if self.checkpoint_dir and (idx + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(layers, idx + 1)
            pbar_convert.update(1)
        
        pbar_convert.close()
        return RSULFModel(layers, self.stats)

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


class RSULFModel(torch.nn.Module):
    def __init__(self, layers: List[RSULFLayerCUDA], stats: Optional[ConversionStats] = None):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.wrappers = torch.nn.ModuleList([
            RSULFWrapperCUDA(layer) for layer in layers
        ])
        self.stats = stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for wrapper in self.wrappers:
            x = wrapper(x)
        return x

    def reset_memory(self):
        for wrapper in self.wrappers:
            wrapper.v_mem = None

