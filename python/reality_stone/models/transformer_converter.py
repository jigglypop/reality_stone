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
        self.calibration_samples = int(max(1, calibration_samples))
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
            weights["ffn_mode"] = "gelu_new"
            if hasattr(layer, 'ln_1'):
                weights["ln_1_weight"] = layer.ln_1.weight.detach().cpu().numpy().astype(np.float32)
                weights["ln_1_bias"] = layer.ln_1.bias.detach().cpu().numpy().astype(np.float32)
            if hasattr(layer, 'ln_2'):
                weights["ln_2_weight"] = layer.ln_2.weight.detach().cpu().numpy().astype(np.float32)
                weights["ln_2_bias"] = layer.ln_2.bias.detach().cpu().numpy().astype(np.float32)
            return weights

        q = layer.self_attn.q_proj
        k = layer.self_attn.k_proj
        v = layer.self_attn.v_proj if hasattr(layer.self_attn, "v_proj") else None
        o = layer.self_attn.o_proj if hasattr(layer.self_attn, "o_proj") else None

        weights["WQ"] = q.weight.detach().cpu().numpy().astype(np.float32)
        weights["WK"] = k.weight.detach().cpu().numpy().astype(np.float32)
        if getattr(q, "bias", None) is not None:
            weights["bQ"] = q.bias.detach().cpu().numpy().astype(np.float32)
        if getattr(k, "bias", None) is not None:
            weights["bK"] = k.bias.detach().cpu().numpy().astype(np.float32)

        if v is not None:
            weights["WV"] = v.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(v, "bias", None) is not None:
                weights["bV"] = v.bias.detach().cpu().numpy().astype(np.float32)
        if o is not None:
            weights["WO"] = o.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(o, "bias", None) is not None:
                weights["bO"] = o.bias.detach().cpu().numpy().astype(np.float32)

        if hasattr(layer.mlp, "gate_proj") and hasattr(layer.mlp, "up_proj") and hasattr(layer.mlp, "down_proj"):
            weights["W1"] = layer.mlp.up_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["W2"] = layer.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["WG"] = layer.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(layer.mlp.gate_proj, "bias", None) is not None:
                weights["bG"] = layer.mlp.gate_proj.bias.detach().cpu().numpy().astype(np.float32)
            weights["ffn_mode"] = "swiglu"
        elif hasattr(layer.mlp, "gate_proj") and hasattr(layer.mlp, "down_proj"):
            weights["W1"] = layer.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["W2"] = layer.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["ffn_mode"] = "silu"
        else:
            weights["W1"] = layer.mlp.fc1.weight.detach().cpu().numpy().astype(np.float32)
            weights["W2"] = layer.mlp.fc2.weight.detach().cpu().numpy().astype(np.float32)
            weights["ffn_mode"] = "gelu"

        if hasattr(layer, "input_layernorm"):
            weights["ln_1_weight"] = layer.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(layer.input_layernorm, "bias", None) is not None:
                weights["ln_1_bias"] = layer.input_layernorm.bias.detach().cpu().numpy().astype(np.float32)
            weights["norm_mode"] = "rmsnorm"
        elif hasattr(layer, "ln_1"):
            weights["ln_1_weight"] = layer.ln_1.weight.detach().cpu().numpy().astype(np.float32)
            weights["ln_1_bias"] = layer.ln_1.bias.detach().cpu().numpy().astype(np.float32)
            weights["norm_mode"] = "layernorm"

        if hasattr(layer, "post_attention_layernorm"):
            weights["ln_2_weight"] = layer.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(layer.post_attention_layernorm, "bias", None) is not None:
                weights["ln_2_bias"] = layer.post_attention_layernorm.bias.detach().cpu().numpy().astype(np.float32)
        elif hasattr(layer, "ln_2"):
            weights["ln_2_weight"] = layer.ln_2.weight.detach().cpu().numpy().astype(np.float32)
            weights["ln_2_bias"] = layer.ln_2.bias.detach().cpu().numpy().astype(np.float32)

        return weights

    def verify_weights(self, weights: Dict[str, np.ndarray], idx: int) -> Tuple[bool, Dict]:
        result = {"valid": True, "issues": []}
        for name, w in weights.items():
            if not isinstance(w, np.ndarray):
                continue
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
                norm_mode=str(weights.get("norm_mode", "layernorm")),
                ffn_mode=str(weights.get("ffn_mode", "gelu")),
                use_fast=bool((not self.exact) and (self.calibration_samples > 1)),
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
            if "WG" in weights:
                rsulf.set_ffn_gate(weights["WG"], weights.get("bG"))
            if "ln_1_weight" in weights:
                rsulf.set_ln1(weights["ln_1_weight"], weights.get("ln_1_bias"))
            if "ln_2_weight" in weights:
                rsulf.set_ln2(weights["ln_2_weight"], weights.get("ln_2_bias"))
            
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
                        use_fast=False,
                        norm_mode=str(weights.get("norm_mode", "layernorm")),
                        ffn_mode=str(weights.get("ffn_mode", "gelu")),
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
                    norm_mode=str(weights.get("norm_mode", "layernorm")),
                    ffn_mode=str(weights.get("ffn_mode", "gelu")),
                    use_fast=bool(self.calibration_samples > 1),
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
                if "WG" in weights:
                    rsulf.set_ffn_gate(weights["WG"], weights.get("bG"))
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

    def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
        for wrapper in self.wrappers:
            x_t = wrapper.forward_step(x_t)
        return x_t

    def reset_memory(self):
        for wrapper in self.wrappers:
            wrapper.v_mem = None

    def init_step_cache(self, batch: int, max_len: int, device: torch.device, dtype: torch.dtype):
        for wrapper in self.wrappers:
            if hasattr(wrapper, "init_step_cache"):
                wrapper.init_step_cache(batch, max_len, device, dtype)


class TorchRiemannianDecoder(nn.Module):
    def __init__(self, u: np.ndarray, a: np.ndarray, bt: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.u = nn.Parameter(torch.from_numpy(np.asarray(u, dtype=np.float32)), requires_grad=False)
        self.a = nn.Parameter(torch.from_numpy(np.asarray(a, dtype=np.float32)), requires_grad=False)
        self.bt = nn.Parameter(torch.from_numpy(np.asarray(bt, dtype=np.float32)), requires_grad=False)
        self.bias = nn.Parameter(torch.from_numpy(np.asarray(bias, dtype=np.float32)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            b, s, d = x.shape
            flat = x.reshape(-1, d).to(dtype=torch.float32)
            y = flat @ self.u
            y = y @ self.a.T
            y = y @ self.bt.T
            y = y + self.bias.unsqueeze(0)
            return y.view(b, s, -1)
        if x.dim() == 2:
            flat = x.to(dtype=torch.float32)
            y = flat @ self.u
            y = y @ self.a.T
            y = y @ self.bt.T
            y = y + self.bias.unsqueeze(0)
            return y
        raise ValueError("decoder input must be 2D or 3D")


class SyntaxHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))
        return x + self.fc2(h)


class RSULFCausalLM(nn.Module):
    def __init__(
        self,
        rsulf: RSULFModel,
        token_embedding: nn.Embedding,
        lm_head: nn.Linear,
        final_norm: Optional[nn.Module] = None,
        pos_embedding: Optional[nn.Embedding] = None,
        decoder=None,
        apply_final_norm: bool = True,
    ):
        super().__init__()
        self.rsulf = rsulf
        self.token_embedding = token_embedding
        self.pos_embedding = pos_embedding
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.decoder = decoder
        self.apply_final_norm = bool(apply_final_norm)
        self.syntax_head = SyntaxHead(int(token_embedding.weight.size(1)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.rsulf, "reset_memory"):
            self.rsulf.reset_memory()
        x = self.token_embedding(input_ids)
        if self.pos_embedding is not None:
            pos = torch.arange(input_ids.size(1), device=input_ids.device, dtype=torch.long)
            x = x + self.pos_embedding(pos)[None, :, :]
        x = self.rsulf(x)
        x = self.syntax_head(x)
        if self.decoder is not None:
            x_np = x.detach().to("cpu", dtype=torch.float32).reshape(-1, x.size(-1)).numpy().astype(np.float32)
            y_np = self.decoder.forward(x_np)
            x = torch.from_numpy(y_np).to(device=input_ids.device, dtype=x.dtype).view(input_ids.size(0), input_ids.size(1), -1)
        if self.final_norm is not None and bool(self.apply_final_norm):
            x = self.final_norm(x)
        return self.lm_head(x)

    def _decode_hidden(self, h: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            return h
        if isinstance(self.decoder, nn.Module):
            y = self.decoder(h)
            return y.to(dtype=h.dtype)
        h_np = h.detach().to("cpu", dtype=torch.float32).reshape(-1, h.size(-1)).numpy().astype(np.float32)
        y_np = self.decoder.forward(h_np)
        y = torch.from_numpy(y_np).to(device=h.device, dtype=h.dtype).view(h.size(0), h.size(1), -1)
        return y

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        return self.generate_sample(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.15,
        )

    def generate_sample(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        from reality_stone.utils.sampling import sample_next_token
        device = input_ids.device
        out = input_ids
        self.rsulf.reset_memory()
        self.rsulf.init_step_cache(
            batch=int(out.size(0)),
            max_len=int(out.size(1) + max(1, int(max_new_tokens)) + 1),
            device=device,
            dtype=self.token_embedding.weight.dtype,
        )
        for pos_idx in range(out.size(1)):
            tok = out[:, pos_idx : pos_idx + 1]
            x_t = self.token_embedding(tok)
            if self.pos_embedding is not None:
                pos = torch.tensor([pos_idx], device=device, dtype=torch.long)
                x_t = x_t + self.pos_embedding(pos)[None, :, :]
            x_t = self.rsulf.forward_step(x_t)

        finished = torch.zeros(out.size(0), device=device, dtype=torch.bool)
        for step in range(int(max_new_tokens)):
            if out.size(1) == 0:
                break
            if step == 0:
                h = x_t
            else:
                tok = out[:, -1:]
                x_t = self.token_embedding(tok)
                if self.pos_embedding is not None:
                    pos = torch.tensor([out.size(1) - 1], device=device, dtype=torch.long)
                    x_t = x_t + self.pos_embedding(pos)[None, :, :]
                x_t = self.rsulf.forward_step(x_t)
                h = x_t
            h = self.syntax_head(h)
            h = self._decode_hidden(h)
            if self.final_norm is not None and bool(self.apply_final_norm):
                h = self.final_norm(h)
            logits = self.lm_head(h)[:, -1, :]
            next_id = sample_next_token(
                logits,
                generated_ids=out,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
            )
            if eos_token_id is not None:
                eos = torch.full_like(next_id, int(eos_token_id))
                next_id = torch.where(finished.unsqueeze(1), eos, next_id)
            out = torch.cat([out, next_id], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_id.squeeze(1) == int(eos_token_id))
                if bool(finished.all().item()):
                    break
        return out


def save_rsulf_causal_lm(path: str, rs_lm: RSULFCausalLM, decoder_state: dict | None = None) -> None:
    state = rs_lm.state_dict()
    state_cpu = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            state_cpu[k] = v.detach().cpu()
        else:
            state_cpu[k] = v

    layers = getattr(getattr(rs_lm, "rsulf", None), "layers", None)
    layer_meta = []
    if layers is not None:
        for layer in layers:
            layer_meta.append(
                {
                    "d_model": int(getattr(layer, "d_model", 0)),
                    "r": int(getattr(layer, "r", 0)),
                    "ffn_dim": int(getattr(getattr(layer, "b1", None), "numel", lambda: 0)()),
                    "num_heads": int(getattr(layer, "num_heads", 1)),
                    "pfc_mode": str(getattr(layer, "pfc_mode", "accel")),
                    "pfc_curvature": float(getattr(layer, "pfc_curvature", 0.0)),
                    "pfc_max_rel": float(getattr(layer, "pfc_max_rel", 0.02)),
                    "pfc_window": int(getattr(layer, "pfc_window", 0)),
                    "pfc_speed_gate": float(getattr(layer, "pfc_speed_gate", 1.0)),
                    "norm_mode": str(getattr(layer, "norm_mode", "layernorm")),
                    "ffn_mode": str(getattr(layer, "ffn_mode", "gelu")),
                }
            )

    token_emb = getattr(rs_lm, "token_embedding", None)
    pos_emb = getattr(rs_lm, "pos_embedding", None)
    meta = {
        "vocab_size": int(token_emb.weight.size(0)) if token_emb is not None else 0,
        "d_model": int(token_emb.weight.size(1)) if token_emb is not None else 0,
        "max_positions": int(pos_emb.weight.size(0)) if pos_emb is not None else 0,
        "num_layers": int(len(layer_meta)),
        "layer_meta": layer_meta,
        "apply_final_norm": bool(getattr(rs_lm, "apply_final_norm", True)),
    }

    payload = {"meta": meta, "state_dict": state_cpu}
    if decoder_state is not None:
        payload["decoder_state"] = {
            "u": np.asarray(decoder_state["u"], dtype=np.float32),
            "a": np.asarray(decoder_state["a"], dtype=np.float32),
            "bt": np.asarray(decoder_state["bt"], dtype=np.float32),
            "bias": np.asarray(decoder_state["bias"], dtype=np.float32),
        }
    torch.save(payload, path)


def load_rsulf_causal_lm(path: str, device: str | torch.device | None = None) -> RSULFCausalLM:
    payload = torch.load(path, map_location="cpu")
    meta = payload.get("meta") or {}
    state = payload.get("state_dict") or {}
    layer_meta = meta.get("layer_meta") or []

    vocab_size = int(meta.get("vocab_size") or 0)
    d_model = int(meta.get("d_model") or 0)
    max_positions = int(meta.get("max_positions") or 0)

    if vocab_size <= 0 or d_model <= 0 or max_positions <= 0:
        raise ValueError("Invalid checkpoint meta (vocab_size/d_model/max_positions)")
    if not layer_meta:
        raise ValueError("Invalid checkpoint meta (layer_meta missing)")

    layers: list[RSULFLayerCUDA] = []
    for lm in layer_meta:
        dm = int(lm.get("d_model") or d_model)
        r = int(lm.get("r") or dm)
        ffn_dim = int(lm.get("ffn_dim") or (4 * dm))
        wq0 = np.zeros((dm, dm), dtype=np.float32)
        wk0 = np.zeros((dm, dm), dtype=np.float32)
        w10 = np.zeros((ffn_dim, dm), dtype=np.float32)
        w20 = np.zeros((dm, ffn_dim), dtype=np.float32)
        layer = RSULFLayerCUDA(
            wq=wq0,
            wk=wk0,
            w1=w10,
            w2=w20,
            d_model=dm,
            r=r,
            eta=0.0,
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            seq_len=0,
            window=0,
            global_basis=None,
            original_block=None,
            use_fast=False,
            calibration_samples=0,
            num_heads=int(lm.get("num_heads") or 1),
            pfc_mode=str(lm.get("pfc_mode") or "accel"),
            pfc_curvature=float(lm.get("pfc_curvature") or 0.0),
            pfc_max_rel=float(lm.get("pfc_max_rel") or 0.02),
            pfc_window=int(lm.get("pfc_window") or 0),
            pfc_speed_gate=float(lm.get("pfc_speed_gate") or 1.0),
            norm_mode=str(lm.get("norm_mode") or "layernorm"),
            ffn_mode=str(lm.get("ffn_mode") or "gelu"),
        )
        layers.append(layer)

    rsulf = RSULFModel(layers, stats=None)

    token_embedding = nn.Embedding(vocab_size, d_model)
    pos_embedding = nn.Embedding(max_positions, d_model)
    final_norm = nn.LayerNorm(d_model, elementwise_affine=True)
    lm_head = nn.Linear(d_model, vocab_size, bias=False)

    rs_lm = RSULFCausalLM(
        rsulf=rsulf,
        token_embedding=token_embedding,
        lm_head=lm_head,
        final_norm=final_norm,
        pos_embedding=pos_embedding,
        decoder=None,
        apply_final_norm=bool(meta.get("apply_final_norm", True)),
    )
    rs_lm.load_state_dict(state, strict=False)

    decoder_state = payload.get("decoder_state")
    if decoder_state is not None:
        rs_lm.decoder = TorchRiemannianDecoder(
            np.asarray(decoder_state["u"], dtype=np.float32),
            np.asarray(decoder_state["a"], dtype=np.float32),
            np.asarray(decoder_state["bt"], dtype=np.float32),
            np.asarray(decoder_state["bias"], dtype=np.float32),
        )

    if device is not None:
        rs_lm = rs_lm.to(device)
    rs_lm.eval()
    return rs_lm


def build_rsulf_causal_lm(model: nn.Module, converter: RSULFTransformerConverter) -> RSULFCausalLM:
    rsulf = converter.convert_model(model)
    return wrap_rsulf_as_causal_lm(model, rsulf)

def wrap_rsulf_as_causal_lm(model: nn.Module, rsulf: RSULFModel) -> RSULFCausalLM:
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        wte = model.transformer.wte
        token_embedding = nn.Embedding(wte.weight.size(0), wte.weight.size(1))
        token_embedding.weight.data = wte.weight.detach().clone().cpu()
        token_embedding.weight.requires_grad = False
        pos_embedding = None
        if hasattr(model.transformer, "wpe"):
            wpe = model.transformer.wpe
            pos_embedding = nn.Embedding(wpe.weight.size(0), wpe.weight.size(1))
            pos_embedding.weight.data = wpe.weight.detach().clone().cpu()
            pos_embedding.weight.requires_grad = False
        final_norm = None
        if hasattr(model.transformer, "ln_f"):
            ln_f = model.transformer.ln_f
            final_norm = nn.LayerNorm(ln_f.weight.numel(), elementwise_affine=True)
            final_norm.weight.data = ln_f.weight.detach().clone().cpu()
            final_norm.bias.data = ln_f.bias.detach().clone().cpu()
            final_norm.weight.requires_grad = False
            final_norm.bias.requires_grad = False
        vocab = token_embedding.weight.size(0)
        d_model = token_embedding.weight.size(1)
        lm_head = nn.Linear(d_model, vocab, bias=False)
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            lm_head.weight.data = model.lm_head.weight.detach().clone().cpu()
        else:
            lm_head.weight.data = token_embedding.weight.detach().clone().cpu()
        lm_head.weight.requires_grad = False
        return RSULFCausalLM(rsulf, token_embedding, lm_head, final_norm=final_norm, pos_embedding=pos_embedding)

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        emb = model.model.embed_tokens
        token_embedding = nn.Embedding(emb.weight.size(0), emb.weight.size(1))
        token_embedding.weight.data = emb.weight.detach().clone().cpu()
        token_embedding.weight.requires_grad = False
        final_norm = None
        if hasattr(model.model, "norm"):
            nrm = model.model.norm
            d_model = token_embedding.weight.size(1)
            ln = nn.LayerNorm(d_model, elementwise_affine=True)
            ln.weight.data = nrm.weight.detach().clone().cpu()
            ln.bias.data.zero_()
            ln.weight.requires_grad = False
            ln.bias.requires_grad = False
            final_norm = ln
        vocab = token_embedding.weight.size(0)
        d_model = token_embedding.weight.size(1)
        lm_head = nn.Linear(d_model, vocab, bias=False)
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            lm_head.weight.data = model.lm_head.weight.detach().clone().cpu()
        else:
            lm_head.weight.data = token_embedding.weight.detach().clone().cpu()
        lm_head.weight.requires_grad = False
        return RSULFCausalLM(rsulf, token_embedding, lm_head, final_norm=final_norm, pos_embedding=None)

    raise ValueError("Unsupported model structure for RSULF causal LM")


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