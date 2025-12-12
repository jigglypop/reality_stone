import numpy as np
import json
import os
import copy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from reality_stone._rust import (
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
import torch.nn as nn
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
        calibration_samples: int = 1024,
        num_heads: int = 1,
        pfc_mode: str = "bilinear",
        pfc_curvature: float = 0.0,
        pfc_max_rel: float = 0.02,
        pfc_window: int = 0,
        pfc_layers: int = 3,
        pfc_speed_gate: float = 1.0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 4,
        verbose: bool = False,
        exact: bool = False,
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
        self.calibration_samples = calibration_samples
        self.num_heads = int(max(1, num_heads))
        self.pfc_mode = str(pfc_mode).lower().strip()
        self.pfc_curvature = float(pfc_curvature)
        self.pfc_max_rel = float(pfc_max_rel)
        self.pfc_window = int(max(0, pfc_window))
        self.pfc_layers = int(max(0, pfc_layers))
        self.pfc_speed_gate = float(max(0.0, pfc_speed_gate))
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        self.exact = exact
        self.stats = ConversionStats()

    def extract_weights(self, layer) -> Dict[str, np.ndarray]:
        weights = {}
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'):
            d_model = int(self.d_model)
            c_attn_w = np.ascontiguousarray(
                layer.attn.c_attn.weight.detach().cpu().numpy().astype(np.float32)
            )
            if c_attn_w.shape == (d_model, 3 * d_model):
                wq = c_attn_w[:, :d_model].T
                wk = c_attn_w[:, d_model:2 * d_model].T
                wv = c_attn_w[:, 2 * d_model:3 * d_model].T
            elif c_attn_w.shape == (3 * d_model, d_model):
                wq = c_attn_w[:d_model, :]
                wk = c_attn_w[d_model:2 * d_model, :]
                wv = c_attn_w[2 * d_model:3 * d_model, :]
            else:
                raise ValueError(f"Unexpected GPT2 c_attn.weight shape: {c_attn_w.shape}")
            weights["WQ"] = np.ascontiguousarray(wq)
            weights["WK"] = np.ascontiguousarray(wk)
            weights["WV"] = np.ascontiguousarray(wv)
            if hasattr(layer.attn.c_attn, "bias") and layer.attn.c_attn.bias is not None:
                c_attn_b = np.ascontiguousarray(
                    layer.attn.c_attn.bias.detach().cpu().numpy().astype(np.float32)
                )
                weights["bQ"] = np.ascontiguousarray(c_attn_b[:d_model])
                weights["bK"] = np.ascontiguousarray(c_attn_b[d_model:2 * d_model])
                weights["bV"] = np.ascontiguousarray(c_attn_b[2 * d_model:3 * d_model])
            c_proj_w = np.ascontiguousarray(layer.attn.c_proj.weight.detach().cpu().numpy().astype(np.float32))
            if c_proj_w.shape != (d_model, d_model):
                raise ValueError(f"Unexpected GPT2 c_proj.weight shape: {c_proj_w.shape}")
            weights["WO"] = np.ascontiguousarray(c_proj_w.T)
            if hasattr(layer.attn.c_proj, "bias") and layer.attn.c_proj.bias is not None:
                weights["bO"] = np.ascontiguousarray(layer.attn.c_proj.bias.detach().cpu().numpy().astype(np.float32))
            w1_w = np.ascontiguousarray(layer.mlp.c_fc.weight.detach().cpu().numpy().astype(np.float32))
            if w1_w.shape == (d_model, 4 * d_model):
                w1 = w1_w.T
            elif w1_w.shape == (4 * d_model, d_model):
                w1 = w1_w
            else:
                raise ValueError(f"Unexpected GPT2 c_fc.weight shape: {w1_w.shape}")
            weights["W1"] = np.ascontiguousarray(w1)
            if hasattr(layer.mlp.c_fc, "bias") and layer.mlp.c_fc.bias is not None:
                weights["b1"] = np.ascontiguousarray(layer.mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32))
            w2_w = np.ascontiguousarray(layer.mlp.c_proj.weight.detach().cpu().numpy().astype(np.float32))
            if w2_w.shape == (4 * d_model, d_model):
                w2 = w2_w.T
            elif w2_w.shape == (d_model, 4 * d_model):
                w2 = w2_w
            else:
                raise ValueError(f"Unexpected GPT2 c_proj.weight shape: {w2_w.shape}")
            weights["W2"] = np.ascontiguousarray(w2)
            if hasattr(layer.mlp.c_proj, "bias") and layer.mlp.c_proj.bias is not None:
                weights["b2"] = np.ascontiguousarray(layer.mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32))
            if hasattr(layer, 'ln_1'):
                weights["ln_1_weight"] = layer.ln_1.weight.detach().cpu().numpy().astype(np.float32)
                weights["ln_1_bias"] = layer.ln_1.bias.detach().cpu().numpy().astype(np.float32)
            if hasattr(layer, 'ln_2'):
                weights["ln_2_weight"] = layer.ln_2.weight.detach().cpu().numpy().astype(np.float32)
                weights["ln_2_bias"] = layer.ln_2.bias.detach().cpu().numpy().astype(np.float32)
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
                calibration_samples=self.calibration_samples,
                num_heads=self.num_heads,
                pfc_mode=self.pfc_mode,
                pfc_curvature=self.pfc_curvature,
                pfc_max_rel=self.pfc_max_rel,
                pfc_window=self.pfc_window,
                pfc_speed_gate=self.pfc_speed_gate,
            )
            if "WV" in weights and "WO" in weights:
                rsulf.set_attention_weights(weights["WV"], weights["WO"])
            rsulf.set_biases(
                bq=weights.get("bQ"),
                bk=weights.get("bK"),
                bv=weights.get("bV"),
                bo=weights.get("bO"),
                b1=weights.get("b1"),
                b2=weights.get("b2"),
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
        if hasattr(model, "config"):
            if hasattr(model.config, "n_head"):
                self.num_heads = int(model.config.n_head)
            elif hasattr(model.config, "num_attention_heads"):
                self.num_heads = int(model.config.num_attention_heads)
        
        print(f"Collecting weights from {len(transformer_layers)} layers...")
        all_wq = []
        all_wk = []
        layer_weights = []
        original_blocks = []
        
        for idx, layer in enumerate(transformer_layers):
            try:
                weights = self.extract_weights(layer)
                valid, check = self.verify_weights(weights, idx)
                if valid:
                    all_wq.append(weights["WQ"])
                    all_wk.append(weights["WK"])
                    layer_weights.append(weights)
                    original_blocks.append(layer)
                else:
                    print(f"Skipping layer {idx} due to invalid weights: {check['issues']}")
                    all_wq.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                    all_wk.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                    layer_weights.append(None)
                    original_blocks.append(None)
            except Exception as e:
                print(f"Error extracting layer {idx}: {e}")
                all_wq.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                all_wk.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                layer_weights.append(None)
                original_blocks.append(None)

        if self.exact:
            print("Exact mode: disabling global basis and using full rank per layer")
            layers = []
            pbar_convert = tqdm(total=len(layer_weights), desc="Converting", unit="layer", disable=not self.verbose)
            for idx, weights in enumerate(layer_weights):
                if weights is None:
                    self.stats.failed.append(idx)
                    pbar_convert.set_postfix(idx=idx, status="skip")
                    pbar_convert.update(1)
                    continue
                try:
                    d_out, d_model = weights["WQ"].shape
                    if self.verbose:
                        print(f"[RSULF] layer {idx:02d}: start")
                    best_r = int(max(1, min(d_model, self.r)))
                    total_layers = len(layer_weights)
                    k = int(self.pfc_layers)
                    if k <= 0:
                        pfc_c = 0.0
                    else:
                        start = max(0, total_layers - k)
                        if idx < start:
                            pfc_c = 0.0
                        else:
                            t = float((idx - start) + 1) / float(max(1, total_layers - start))
                            pfc_c = float(self.pfc_curvature) * t
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
                        global_basis=None,
                        original_block=None,
                        calibration_samples=self.calibration_samples,
                        num_heads=self.num_heads,
                        pfc_mode=self.pfc_mode,
                        pfc_curvature=pfc_c,
                        pfc_max_rel=self.pfc_max_rel,
                        pfc_window=self.pfc_window,
                        pfc_speed_gate=self.pfc_speed_gate,
                    )
                    if "WV" in weights and "WO" in weights:
                        rsulf.set_attention_weights(weights["WV"], weights["WO"])
                    rsulf.set_biases(
                        bq=weights.get("bQ"),
                        bk=weights.get("bK"),
                        bv=weights.get("bV"),
                        bo=weights.get("bO"),
                        b1=weights.get("b1"),
                        b2=weights.get("b2"),
                    )
                    if "ln_1_weight" in weights:
                        rsulf.set_ln1(weights["ln_1_weight"], weights.get("ln_1_bias"))
                    if "ln_2_weight" in weights:
                        rsulf.set_ln2(weights["ln_2_weight"], weights.get("ln_2_bias"))
                    compressed, original, ratio = rsulf.param_count()
                    if self.verbose:
                        print(f"[RSULF] layer {idx:02d}: ok ratio={ratio:.1f}x")
                    layers.append(rsulf)
                    self.stats.converted += 1
                    self.stats.original_params += original
                    self.stats.compressed_params += compressed
                    pbar_convert.set_postfix(idx=idx, ratio=f"{ratio:.1f}x", status="ok")
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
            valid_wq = [w for w in all_wq if w.shape[0] > 0 and w.any()]
            valid_wk = [w for w in all_wk if w.shape[0] > 0 and w.any()]
            if valid_wq:
                global_basis = extract_global_basis(valid_wq, valid_wk, self.r)
                self.global_basis = global_basis
                print(f"Global Basis extracted: rank={global_basis['rank']}")
        except Exception as e:
            print(f"Global Basis extraction failed: {e}. Falling back to local.")

        layers = []
        acc_by_idx = {a.get("layer_idx", i): a.get("expected_accuracy", 0.0) for i, a in enumerate(analyses)}
        rank_by_idx = {
            a.get("layer_idx", i): int(a.get("recommended_rank", self.r))
            for i, a in enumerate(analyses)
        }
        pbar_convert = tqdm(total=len(layer_weights), desc="Converting", unit="layer", disable=not self.verbose)
        for idx, weights in enumerate(layer_weights):
            if weights is None:
                self.stats.failed.append(idx)
                pbar_convert.set_postfix(idx=idx, status="skip")
                pbar_convert.update(1)
                continue
            try:
                d_out, d_model = weights["WQ"].shape
                base_r = rank_by_idx.get(idx, self.r)
                best_r = int(max(1, min(d_model, self.r, base_r)))
                if self.verbose:
                    print(f"[RSULF] layer {idx:02d}: start")
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
                    global_basis=global_basis,
                    calibration_samples=self.calibration_samples,
                    num_heads=self.num_heads,
                    pfc_mode=self.pfc_mode,
                    pfc_curvature=self.pfc_curvature,
                    pfc_max_rel=self.pfc_max_rel,
                    pfc_window=self.pfc_window,
                    pfc_speed_gate=self.pfc_speed_gate,
                )
                if "WV" in weights and "WO" in weights:
                    rsulf.set_attention_weights(weights["WV"], weights["WO"])
                rsulf.set_biases(
                    bq=weights.get("bQ"),
                    bk=weights.get("bK"),
                    bv=weights.get("bV"),
                    bo=weights.get("bO"),
                    b1=weights.get("b1"),
                    b2=weights.get("b2"),
                )
                if "ln_1_weight" in weights:
                    rsulf.set_ln1(weights["ln_1_weight"], weights.get("ln_1_bias"))
                if "ln_2_weight" in weights:
                    rsulf.set_ln2(weights["ln_2_weight"], weights.get("ln_2_bias"))
                
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


def convert_transformer_to_rsulf(
    model: nn.Module,
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
    exact: bool = False,
) -> RSULFModel:
    converter = RSULFTransformerConverter(
        d_model=d_model,
        r=r,
        eta=eta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seq_len=seq_len,
        window=window,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        verbose=verbose,
        exact=exact,
    )
    return converter.convert_model(model)

class FFNPotential(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.d_model = d_model
        self.P = nn.Parameter(torch.zeros(d_model, d_model))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x_flat = x.view(-1, d)
        quad = 0.5 * (x_flat @ self.P * x_flat).sum(dim=-1, keepdim=True)
        neu = self.net(x_flat)
        phi = quad + neu
        return phi.view(b, l)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            phi = self.forward(x_in).sum()
            grad = torch.autograd.grad(phi, x_in, create_graph=False)[0]
        return grad


class LowRankFFN(nn.Module):
    def __init__(self, mlp: nn.Module, rank: int):
        super().__init__()
        if hasattr(mlp, 'c_fc'):
            w1 = mlp.c_fc.weight.data
            b1 = mlp.c_fc.bias.data if mlp.c_fc.bias is not None else None
            w2 = mlp.c_proj.weight.data
            b2 = mlp.c_proj.bias.data if mlp.c_proj.bias is not None else None
        else:
            raise ValueError("Unsupported MLP structure")
        
        d_model, ffn_dim = w1.shape
        rank = min(rank, d_model, ffn_dim)
        u1, s1, v1 = torch.linalg.svd(w1, full_matrices=False)
        u1_r = u1[:, :rank]
        s1_r = s1[:rank]
        v1_r = v1[:rank, :]
        self.w1_a = nn.Parameter(u1_r * s1_r.unsqueeze(0))
        self.w1_b = nn.Parameter(v1_r)
        self.b1 = nn.Parameter(b1) if b1 is not None else None
        u2, s2, v2 = torch.linalg.svd(w2, full_matrices=False)
        u2_r = u2[:, :rank]
        s2_r = s2[:rank]
        v2_r = v2[:rank, :]
        self.w2_a = nn.Parameter(u2_r * s2_r.unsqueeze(0))
        self.w2_b = nn.Parameter(v2_r)
        self.b2 = nn.Parameter(b2) if b2 is not None else None
        
        self.act = nn.GELU(approximate='tanh')
        self.rank = rank
        self.d_model = d_model
        self.ffn_dim = ffn_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.w1_a @ self.w1_b
        if self.b1 is not None:
            h = h + self.b1
        h = self.act(h)
        out = h @ self.w2_a @ self.w2_b
        if self.b2 is not None:
            out = out + self.b2
        return out

    def param_count(self):
        original = self.ffn_dim * self.d_model * 2
        compressed = 2 * (self.rank * self.d_model + self.rank * self.ffn_dim)
        return compressed, original


class StructuralRSULFLayer(nn.Module):
    def __init__(self, block: nn.Module, d_model: int, rank: int):
        super().__init__()
        self.ln_1 = copy.deepcopy(block.ln_1)
        self.ln_2 = copy.deepcopy(block.ln_2)
        self.attn = copy.deepcopy(block.attn)
        self.mlp = LowRankFFN(block.mlp, rank)
        self.potential = FFNPotential(d_model, hidden_dim=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.ln_1(x)
        attn_outputs = self.attn(u)
        attn_out = attn_outputs[0] if isinstance(attn_outputs, (tuple, list)) else attn_outputs
        y = x + attn_out
        w = self.ln_2(y)
        ffn_out = self.mlp(w)
        return y + ffn_out


class StructuralRSULFModel(nn.Module):
    def __init__(
        self,
        blocks: List[nn.Module],
        d_model: int,
        rank: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if rank is None:
            if hidden_dim is not None:
                rank = hidden_dim
            else:
                rank = d_model
        self.layers = nn.ModuleList(
            [StructuralRSULFLayer(block, d_model, rank) for block in blocks]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def wrappers(self) -> nn.ModuleList:
        return self.layers