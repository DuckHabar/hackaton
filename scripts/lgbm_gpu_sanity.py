"""
LightGBM GPU sanity check перед EXP-002.
также проверяет наличие GPU build (если CPU only, печатает warning).
"""
import time
import numpy as np
import lightgbm as lgb

X = np.random.rand(50_000, 50).astype(np.float32)
y = np.random.rand(50_000).astype(np.float32)

train_set = lgb.Dataset(X, label=y)
print("LightGBM version:", lgb.__version__)

# Сначала CPU baseline
print("--- CPU fit (50k × 50, 200 rounds) ---")
t0 = time.time()
model_cpu = lgb.train(
    {"objective": "quantile", "alpha": 0.5, "learning_rate": 0.1,
     "num_leaves": 63, "verbose": -1, "device": "cpu"},
    train_set, num_boost_round=200,
)
print(f"CPU time: {time.time() - t0:.2f}s")

# GPU try
print("--- GPU fit attempt ---")
try:
    t0 = time.time()
    model_gpu = lgb.train(
        {"objective": "quantile", "alpha": 0.5, "learning_rate": 0.1,
         "num_leaves": 63, "verbose": -1, "device": "gpu"},
        train_set, num_boost_round=200,
    )
    print(f"GPU time: {time.time() - t0:.2f}s, GPU build OK")
except Exception as e:
    print(f"GPU build NOT available: {e}")
    print("EXP-002 will use CPU (32 threads should be fine for 32k rows)")

# Также torch GPU warmup для занятости
try:
    import torch
    if torch.cuda.is_available():
        print(f"\nPyTorch CUDA: {torch.cuda.get_device_name(0)}")
        a = torch.randn(8192, 8192, device="cuda")
        for _ in range(50):
            _ = a @ a
        torch.cuda.synchronize()
        print("Torch matmul 8192² × 50 done")
except Exception as e:
    print(f"Torch err: {e}")
