import os
import time
import inspect
import torch

if torch.cuda.is_available():
    print(f"[intent_clinc] CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    print("[intent_clinc] CUDA is NOT available. Training will be slow.")

os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache\transformers"

print("[intent_clinc] Importing HuggingFace Datasets ...")
from datasets import load_dataset

print("[intent_clinc] Importing Transformers (first time can take a while) ...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

print("[intent_clinc] Importing sklearn / numpy ...")
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

MODEL_NAME = "microsoft/deberta-v3-base"
DATASET_NAME = "clinc_oos"
DATASET_CONFIG = "plus"

print("[intent_clinc] Loading dataset:", DATASET_NAME, DATASET_CONFIG)
start = time.time()
ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
print(f"[intent_clinc] Dataset loaded in {time.time() - start:.1f}s")
num_labels = ds["train"].features["intent"].num_classes

print("[intent_clinc] Loading tokenizer/model:", MODEL_NAME)
start = time.time()
# use_fast=False 로 설정해 tiktoken 의존성을 피하고, 순수 SentencePiece 토크나이저 사용
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def preprocess(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    enc["labels"] = examples["intent"]
    return enc

print("[intent_clinc] Tokenizing dataset ...")
start = time.time()
encoded = ds.map(preprocess, batched=True)
print(f"[intent_clinc] Tokenization finished in {time.time() - start:.1f}s")

print("[intent_clinc] Loading classification head ...")
start = time.time()
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
)
print(f"[intent_clinc] Model ready in {time.time() - start:.1f}s")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

output_dir = "clinc_deberta_v3_base"
os.makedirs(output_dir, exist_ok=True)

ta_params = inspect.signature(TrainingArguments.__init__).parameters
ta_kwargs = dict(
    output_dir=output_dir,
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
    no_cuda=False,
)
has_eval = "evaluation_strategy" in ta_params
has_save = "save_strategy" in ta_params

if torch.cuda.is_available() and "fp16" in ta_params:
    ta_kwargs["fp16"] = True
    print("[intent_clinc] CUDA detected, enabling fp16 training")

if has_eval:
    ta_kwargs["evaluation_strategy"] = "epoch"
if has_save:
    ta_kwargs["save_strategy"] = "epoch"
if "save_total_limit" in ta_params:
    ta_kwargs["save_total_limit"] = 2
if "warmup_ratio" in ta_params:
    ta_kwargs["warmup_ratio"] = 0.06

# load_best_model_at_end 는 eval/save 둘 다 지원할 때만 켠다.
if has_eval and has_save and "load_best_model_at_end" in ta_params:
    ta_kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in ta_params:
        ta_kwargs["metric_for_best_model"] = "accuracy"

if "disable_tqdm" in ta_params:
    ta_kwargs["disable_tqdm"] = False

print("[intent_clinc] TrainingArguments kwargs:", ta_kwargs)
args = TrainingArguments(**ta_kwargs)

print("[intent_clinc] Initializing Trainer ...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    compute_metrics=compute_metrics,
)

print("[intent_clinc] Starting training ...")
train_start = time.time()
trainer.train()
print(f"[intent_clinc] Training finished in {(time.time() - train_start) / 60:.1f} min")

print("[intent_clinc] Evaluating on test set ...")
metrics = trainer.evaluate(encoded["test"])
print("[intent_clinc] Test metrics:", metrics)

try:
    # torch is already imported at the top
    import _rust
except ImportError:
    print("[intent_clinc] Hyperbolic glue: _rust not available, skipping state encoder definition.")
else:
    def encode_intent_to_hyperbolic_state(
        texts,
        tokenizer,
        model,
        device: str | None = None,
        c: float = 1.0,
        t: float = 1.0,
    ):
        """
        텍스트 → intent 확률분포 → Poincaré 볼 임베딩으로 매핑.

        - logits → softmax(probs)
        - probs 를 u=0, v=probs 로 두고 _rust.poincare_ball_layer_cpu 적용
        - 모델은 가능하면 GPU('cuda')에서, 하이퍼볼릭 연산은 기존 Rust CPU 구현 사용
        """
        if isinstance(texts, str):
            texts = [texts]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)
        model.eval()

        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        for k in enc:
            enc[k] = enc[k].to(device)

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.detach().cpu().numpy().astype("float32")
        import numpy as np

        u = np.zeros_like(probs_np, dtype="float32")
        v = probs_np

        y = _rust.poincare.poincare_ball_layer_cpu(u, v, float(c), float(t))
        y_np = np.array(y, dtype="float32")
        if y_np.shape[0] == 1:
            return y_np[0]
        return y_np