"""
CUDA 최적화 벤치마크

Python vs CUDA 커널 성능 비교
"""
import time
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from reality_stone.layers.metric_attention import MetricAttention, HAS_CUDA_KERNEL

def benchmark_geodesic_attention(
    batch_size: int = 2,
    num_heads: int = 4,
    seq_len: int = 64,
    d_head: int = 16,
    k_topk: int = 8,
    num_runs: int = 100,
    device: str = 'cuda'
):
    """
    Geodesic Top-k Attention 벤치마크
    """
    print(f"\n{'='*60}")
    print(f"Geodesic Top-k Attention Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"CUDA Kernel Available: {HAS_CUDA_KERNEL}")
    print(f"Batch Size: {batch_size}")
    print(f"Num Heads: {num_heads}")
    print(f"Seq Length: {seq_len}")
    print(f"Head Dim: {d_head}")
    print(f"Top-k: {k_topk}")
    print(f"Num Runs: {num_runs}")
    print(f"{'='*60}\n")
    
    # Create model
    model = MetricAttention(
        hidden_size=d_head,
        mode="geodesic",
        manifold="poincare",
        c=1.0,
        normalizer="softmax",
        rank=2
    ).to(device)
    
    # Create dummy inputs
    q = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)
    
    # Create topology index (top-k neighbors)
    topo_idx = torch.randint(0, seq_len, (batch_size, seq_len, k_topk), device=device)
    topk_cfg = {"neighbor": k_topk}
    
    # Metric keys
    metric_keys = ["topic:diagnosis", "priority:high"]
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(q, k, v, topo_idx=topo_idx, topk_cfg=topk_cfg, metric_keys=metric_keys)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {num_runs} iterations...")
    start = time.perf_counter()
    
    for _ in range(num_runs):
        with torch.no_grad():
            out = model(q, k, v, topo_idx=topo_idx, topk_cfg=topk_cfg, metric_keys=metric_keys)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    # Results
    total_time = (end - start) * 1000  # ms
    avg_time = total_time / num_runs
    throughput = (batch_size * seq_len * num_runs) / (total_time / 1000)  # tokens/sec
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Total Time: {total_time:.2f} ms")
    print(f"Avg Time per Iteration: {avg_time:.4f} ms")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"{'='*60}\n")
    
    return avg_time, throughput


def compare_python_vs_cuda():
    """
    Python vs CUDA 커널 비교
    """
    print("\n" + "="*60)
    print("PYTHON vs CUDA COMPARISON")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping comparison")
        return
    
    configs = [
        # (batch_size, num_heads, seq_len, d_head, k_topk)
        (1, 4, 32, 16, 8),
        (2, 4, 64, 16, 8),
        (4, 4, 128, 16, 8),
        (2, 8, 64, 32, 16),
    ]
    
    results = []
    
    for batch_size, num_heads, seq_len, d_head, k_topk in configs:
        print(f"\n{'='*60}")
        print(f"Config: B={batch_size}, H={num_heads}, T={seq_len}, d={d_head}, K={k_topk}")
        print(f"{'='*60}")
        
        # Python baseline (CPU)
        print("\n[1/2] Python (CPU)")
        try:
            time_cpu, tput_cpu = benchmark_geodesic_attention(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                d_head=d_head,
                k_topk=k_topk,
                num_runs=50,
                device='cpu'
            )
        except Exception as e:
            print(f"CPU benchmark failed: {e}")
            time_cpu, tput_cpu = None, None
        
        # CUDA
        print("\n[2/2] CUDA")
        try:
            time_cuda, tput_cuda = benchmark_geodesic_attention(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                d_head=d_head,
                k_topk=k_topk,
                num_runs=100,
                device='cuda'
            )
        except Exception as e:
            print(f"CUDA benchmark failed: {e}")
            time_cuda, tput_cuda = None, None
        
        # Speedup
        if time_cpu and time_cuda:
            speedup = time_cpu / time_cuda
            results.append({
                'config': f"B={batch_size}, H={num_heads}, T={seq_len}",
                'cpu_time': time_cpu,
                'cuda_time': time_cuda,
                'speedup': speedup,
                'cuda_tput': tput_cuda
            })
            print(f"\n🚀 Speedup: {speedup:.2f}x")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Config':<30} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Speedup':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['config']:<30} {r['cpu_time']:<12.4f} {r['cuda_time']:<12.4f} {r['speedup']:<10.2f}x")
    print("="*60)
    
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"\n⚡ Average Speedup: {avg_speedup:.2f}x")


def profile_layers():
    """
    각 레이어별 시간 측정
    """
    print("\n" + "="*60)
    print("LAYER-WISE PROFILING")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    from reality_stone.utils.pre_segmenter import PreSegmenter
    from reality_stone.models.sentence_topic_head import SentenceTopicHead
    from reality_stone.models.metric_router import MetricContextRouter
    from reality_stone.models.rce_lexical_decoder import RCELexicalDecoder
    
    # Config
    d_model = 64
    d_head = 64
    num_topics = 8
    num_heads = 4
    vocab_size = 50000
    device = 'cuda'
    
    # Create modules
    segmenter = PreSegmenter(max_seq_len=24)
    topic_head = SentenceTopicHead(d_model, d_head, num_topics, num_heads).to(device)
    router = MetricContextRouter(d_head, num_topics).to(device)
    decoder = RCELexicalDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=2,
        n_head=num_heads,
        manifold="poincare",
        c=1.0
    ).to(device)
    
    # Sample input
    text = "양자역학은 원자와 분자의 세계를 설명한다. 불확정성 원리가 핵심이다."
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        seg_output = segmenter(text)
        T = seg_output['num_sentences']
        sentence_emb = torch.randn(1, T, d_model, device=device)
        topo_idx = torch.zeros(1, T, 3, dtype=torch.long, device=device)
        
        P_topic, scores, metric_keys = topic_head(sentence_emb, topo_idx)
        L = router(P_topic, scores, metric_keys)
        
        tokens_input = torch.randint(0, vocab_size, (1, T), device=device)
        mask_input = torch.ones(1, T, device=device)
        candidates = {}
        
        output_ids = decoder(tokens_input, mask_input, candidates)
    
    torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 100
    print(f"Running {num_runs} iterations...\n")
    
    times = {
        'L0_segmenter': [],
        'L1_topic_head': [],
        'L2_router': [],
        'L3_decoder': [],
    }
    
    for _ in range(num_runs):
        # L0
        start = time.perf_counter()
        seg_output = segmenter(text)
        times['L0_segmenter'].append((time.perf_counter() - start) * 1000)
        
        T = seg_output['num_sentences']
        sentence_emb = torch.randn(1, T, d_model, device=device)
        topo_idx = torch.zeros(1, T, 3, dtype=torch.long, device=device)
        
        # L1
        torch.cuda.synchronize()
        start = time.perf_counter()
        P_topic, scores, metric_keys = topic_head(sentence_emb, topo_idx)
        torch.cuda.synchronize()
        times['L1_topic_head'].append((time.perf_counter() - start) * 1000)
        
        # L2
        torch.cuda.synchronize()
        start = time.perf_counter()
        L = router(P_topic, scores, metric_keys)
        torch.cuda.synchronize()
        times['L2_router'].append((time.perf_counter() - start) * 1000)
        
        # L3
        tokens_input = torch.randint(0, vocab_size, (1, T), device=device)
        mask_input = torch.ones(1, T, device=device)
        candidates = {}
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        output_ids = decoder(tokens_input, mask_input, candidates)
        torch.cuda.synchronize()
        times['L3_decoder'].append((time.perf_counter() - start) * 1000)
    
    # Results
    print("="*60)
    print(f"{'Layer':<20} {'Avg Time (ms)':<15} {'Percentage':<15}")
    print("-"*60)
    
    total_time = sum(sum(v) for v in times.values())
    
    for layer, time_list in times.items():
        avg_time = sum(time_list) / len(time_list)
        percentage = (sum(time_list) / total_time) * 100
        print(f"{layer:<20} {avg_time:<15.4f} {percentage:<15.2f}%")
    
    print("-"*60)
    print(f"{'Total':<20} {total_time/num_runs:<15.4f} {'100.00%':<15}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA Optimization Benchmark")
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'compare', 'profile'],
                       help='Benchmark mode')
    
    args = parser.parse_args()
    
    if args.mode in ['all', 'compare']:
        compare_python_vs_cuda()
    
    if args.mode in ['all', 'profile']:
        profile_layers()
    
    print("\n✅ Benchmark complete!")

