import argparse
import contextlib
import datetime
import hashlib
import itertools
import json
import logging
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
    set_seed,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _cache_prefix(tokenizer, file_path: str, max_seq_len: int, cache_dir: Optional[str]) -> str:
    """Cache file prefix keyed on the data file, tokenizer and max_seq_len."""
    st = os.stat(file_path)
    vocab_hash = hashlib.md5(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    probe = tokenizer("Hello there")["input_ids"]
    key = hashlib.md5(
        f"{os.path.abspath(file_path)}|{st.st_size}|{int(st.st_mtime)}|{max_seq_len}|"
        f"{vocab_hash}|{probe}".encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(file_path)), "token_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{os.path.basename(file_path)}.L{max_seq_len}.{key}")


def build_token_cache(tokenizer, file_path: str, max_seq_len: int, prefix: str, chunk_lines: int = 100_000):
    """Tokenize every non-empty line once and store the ids as one flat array plus
    per-line offsets (no padding). Written atomically so a killed job never leaves
    a half-written cache behind."""
    dtype = np.uint16 if len(tokenizer) < 2**16 else np.int32
    id_chunks, len_chunks, pending = [], [], []

    def flush():
        enc = tokenizer(pending, truncation=True, max_length=max_seq_len)["input_ids"]
        len_chunks.append(np.fromiter(map(len, enc), dtype=np.int64, count=len(enc)))
        id_chunks.append(np.fromiter(itertools.chain.from_iterable(enc), dtype=dtype))
        pending.clear()

    logger.info(f"[Data] Tokenizing {file_path} (one-off, cached to {prefix}.*)")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pending.append(line)
            if len(pending) >= chunk_lines:
                flush()
                logger.info(f"[Data] tokenized {sum(len(x) for x in len_chunks)} lines")
        if pending:
            flush()

    if not len_chunks:
        raise ValueError(f"No non-empty lines found in {file_path}")
    ids = np.concatenate(id_chunks)
    offsets = np.zeros(sum(len(x) for x in len_chunks) + 1, dtype=np.int64)
    np.cumsum(np.concatenate(len_chunks), out=offsets[1:])

    # offsets last: the cache counts as complete only once both files exist.
    for arr, suffix in ((ids, "ids"), (offsets, "offsets")):
        tmp = f"{prefix}.{suffix}.{os.getpid()}.tmp.npy"  # unique per process
        np.save(tmp, arr)
        os.replace(tmp, f"{prefix}.{suffix}.npy")


def ensure_token_cache(tokenizer, file_path: str, max_seq_len: int, prefix: str,
                       stale_after_s: float = 4 * 3600, poll_s: float = 20.0):
    """Make sure the token cache exists, building it in at most one process.

    The first process to create `<prefix>.lock` (atomic O_EXCL) builds the cache;
    all others poll until both cache files appear and then just reuse them. A lock
    older than `stale_after_s` (builder was killed) is removed and taken over."""
    ids_path, offsets_path = f"{prefix}.ids.npy", f"{prefix}.offsets.npy"
    lock_path = f"{prefix}.lock"
    done = lambda: os.path.isfile(ids_path) and os.path.isfile(offsets_path)

    while not done():
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_after_s:
                    logger.warning(f"[Data] Removing stale cache lock {lock_path}")
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue  # lock vanished: builder finished or gave up, re-check
            logger.info(f"[Data] Another job is building {prefix}.*; waiting...")
            time.sleep(poll_s)
            continue
        try:
            os.write(fd, f"{os.uname().nodename} pid={os.getpid()}\n".encode())
            os.close(fd)
            if not done():  # may have been finished between our check and the lock
                build_token_cache(tokenizer, file_path, max_seq_len, prefix)
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


class TokenizedDataset(Dataset):
    """Pre-tokenized, unpadded line dataset backed by a memory-mapped id array
    (shared between dataloader workers and ranks, no per-process copies)."""

    def __init__(self, tokenizer, file_path: str, max_seq_len: int, cache_dir: Optional[str] = None):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Training file not found: {file_path}")
        prefix = _cache_prefix(tokenizer, file_path, max_seq_len, cache_dir)
        ids_path, offsets_path = f"{prefix}.ids.npy", f"{prefix}.offsets.npy"

        # Only rank 0 of each job touches the cache; the other ranks wait at the
        # barrier. Across concurrent jobs (e.g. a seed array) a lock file makes
        # exactly one job build the cache while the others wait and then reuse it.
        if is_main_process():
            ensure_token_cache(tokenizer, file_path, max_seq_len, prefix)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self.ids = np.load(ids_path, mmap_mode="r")
        self.offsets = np.load(offsets_path)
        self.lengths = np.diff(self.offsets)
        logger.info(
            f"Loaded {len(self.lengths)} examples from {file_path} "
            f"(max_seq_len={max_seq_len}, mean_len={self.lengths.mean():.1f}, "
            f"max_len={self.lengths.max()})"
        )

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return torch.from_numpy(
            np.asarray(self.ids[self.offsets[idx] : self.offsets[idx + 1]], dtype=np.int64)
        )


class PadCollator:
    """Pads each batch only to its longest sequence (right padding, labels -100
    on padding) -- same examples as padding everything to max_seq_len."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, seqs: List[torch.Tensor]) -> dict:
        max_len = max(len(x) for x in seqs)
        input_ids = torch.full((len(seqs), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, x in enumerate(seqs):
            input_ids[i, : len(x)] = x
            attention_mask[i, : len(x)] = 1
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class LengthAwareBatchSampler:
    """Shuffled batch sampler that is distributed-aware (every rank derives the
    same epoch order from seed+epoch and takes its own slice of batches).

    With ``bucket=True``, examples are sorted by length inside large random chunks
    before batching, then the batches are shuffled; this minimises padding."""

    def __init__(self, lengths, batch_size, seed=0, bucket=False, rank=0, world_size=1):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.bucket = bucket
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batches(self):
        n, bs = len(self.lengths), self.batch_size
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(n, generator=g).numpy()
        if self.bucket:
            batches = []
            chunk = bs * 100
            for s in range(0, n, chunk):
                c = perm[s : s + chunk]
                c = c[np.argsort(self.lengths[c], kind="stable")]
                batches += [c[i : i + bs] for i in range(0, len(c), bs)]
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        else:
            batches = [perm[i : i + bs] for i in range(0, n, bs)]
        if self.world_size > 1:
            # Every rank must see the same number of equally sized batches.
            batches = [b for b in batches if len(b) == bs]
            batches = batches[: len(batches) // self.world_size * self.world_size]
            batches = batches[self.rank :: self.world_size]
        return batches

    def __iter__(self):
        for b in self._batches():
            yield b.tolist()

    def __len__(self):
        n, bs = len(self.lengths), self.batch_size
        if self.world_size > 1:
            return (n // bs) // self.world_size
        return math.ceil(n / bs)


def build_model_and_tokenizer(
    tokenizer_path: str,
    lower: bool = False,
    hidden_size: int = 768,
    num_layers: int = 6,
    num_heads: int = 12,
    max_position_embeddings: int = 512,
    attn_implementation: Optional[str] = None,
) -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """Build a decoder-only GPT-2 style model and tokenizer.

    Uses a Hugging Face-compatible tokenizer from ``tokenizer_path``.
    This should point to ``src/tokenizer/tokenizer_char``.
    """
    logger.info(f"[Init] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _ensure_usable_tokenizer(tokenizer, tokenizer_path)

    # Build GPT-2 style config, using your desired dimensions
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=max_position_embeddings,
        n_ctx=max_position_embeddings,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation=attn_implementation,
    )
    if attn_implementation == "flash_attention_3" and config.attn_pdrop > 0:
        logger.warning(
            f"[Init] attn_implementation=flash_attention_3 does not support attention "
            f"dropout; attn_pdrop={config.attn_pdrop} will be silently ignored during "
            "attention (resid/embd dropout are unaffected)."
        )

    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    logger.info(
        f"[Init] Decoder-only model created with hidden_size={hidden_size}, "
        f"num_layers={num_layers}, num_heads={num_heads}, "
        f"max_position_embeddings={max_position_embeddings}, "
        f"vocab_size={tokenizer.vocab_size}"
    )
    logger.info(
        f"[Init] Tokenizer pad_token_id={tokenizer.pad_token_id}, "
        f"bos_token_id={getattr(tokenizer, 'bos_token_id', None)}, "
        f"eos_token_id={getattr(tokenizer, 'eos_token_id', None)}"
    )
    return model, tokenizer


def _ensure_usable_tokenizer(tokenizer, source: str):
    if tokenizer.pad_token is None:
        pad = getattr(tokenizer, "eos_token", None) or getattr(
            tokenizer, "cls_token", None
        )
        if pad is None:
            raise ValueError("Tokenizer has no pad/eos/cls token defined.")
        tokenizer.add_special_tokens({"pad_token": pad})
    vocab_len = len(tokenizer)
    if tokenizer.vocab_size == 0 or vocab_len <= 1:
        raise ValueError(
            f"Tokenizer from {source} is empty "
            f"(vocab_size={tokenizer.vocab_size}, len={vocab_len}, "
            f"pad_token_id={tokenizer.pad_token_id}). Refusing to train; "
            "this produces all-padding batches and NaN loss."
        )


def load_model_and_tokenizer_from_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    attn_implementation: Optional[str] = None,
) -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """Load a saved GPT-2 LM checkpoint; tokenizer always comes from tokenizer_path."""
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    logger.info(f"[Init] Loading model from {checkpoint_path}")
    logger.info(f"[Init] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _ensure_usable_tokenizer(tokenizer, tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(
        checkpoint_path, attn_implementation=attn_implementation
    )
    logger.info(
        f"[Init] Loaded checkpoint: vocab_size={tokenizer.vocab_size}, "
        f"len(tokenizer)={len(tokenizer)}, pad_token_id={tokenizer.pad_token_id}, "
        f"model.config.vocab_size={model.config.vocab_size}"
    )
    return model, tokenizer


def reset_embedding_weights(model: GPT2LMHeadModel) -> None:
    """Re-initialise the word-token embedding layer (wte) and the tied
    output projection (lm_head) to random weights, destroying lexical
    knowledge while keeping positional embeddings and transformer layers
    intact.  Uses the same Normal(0, initializer_range) scheme that GPT-2
    applies at construction time."""
    init_range = model.config.initializer_range  # default 0.02

    # Reset input embeddings
    model.transformer.wte.weight.data.normal_(mean=0.0, std=init_range)
    logger.info(
        f"[ResetEmb] Re-initialised wte embeddings "
        f"(shape={list(model.transformer.wte.weight.shape)}, std={init_range})"
    )

    # If lm_head is weight-tied to wte the above already covers it;
    # if untied, reset it separately.
    if not model.config.tie_word_embeddings:
        model.lm_head.weight.data.normal_(mean=0.0, std=init_range)
        logger.info("[ResetEmb] Re-initialised untied lm_head weights")
    else:
        logger.info("[ResetEmb] lm_head is tied to wte; no separate reset needed")


def create_dataloader(
    file_path: str,
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    num_workers: int,
    seed: int = 0,
    length_bucketing: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, int]:
    """``batch_size`` is per GPU/process."""
    dataset = TokenizedDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        max_seq_len=max_seq_len,
        cache_dir=cache_dir,
    )
    ddp = dist.is_available() and dist.is_initialized()
    batch_sampler = LengthAwareBatchSampler(
        dataset.lengths,
        batch_size=batch_size,
        seed=seed,
        bucket=length_bucketing,
        rank=dist.get_rank() if ddp else 0,
        world_size=dist.get_world_size() if ddp else 1,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=PadCollator(tokenizer.pad_token_id),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataloader, len(dataloader)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def setup_distributed() -> Tuple[int, int, torch.device]:
    """Initialise torch.distributed when launched with torchrun; otherwise single process."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=datetime.timedelta(hours=3),  # rank 0 may tokenize while others wait
        )
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        return dist.get_rank(), dist.get_world_size(), device
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_epoch_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated epoch list. Empty / None -> save every epoch."""
    if value is None or str(value).strip() == "":
        return None
    epochs = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        epochs.append(int(part))
    return epochs or None


def optimizer_steps_per_epoch(num_batches: int, grad_acc: int) -> int:
    """Number of optimizer updates per epoch (one update every ``grad_acc`` batches,
    plus a final flush if the epoch length is not divisible by ``grad_acc``)."""
    grad_acc = max(int(grad_acc), 1)
    return math.ceil(num_batches / grad_acc)


def train_one_phase(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    grad_acc: int,
    logging_steps: int,
    device: torch.device,
    phase_name: str,
    save_dir: str = None,
    tokenizer=None,
    save_epochs: Optional[List[int]] = None,
    seed: int = 0,
    bf16: bool = False,
):
    """
    Train for exactly ``num_epochs`` passes over ``dataloader``.

    ``global_step`` / the LR schedule count **optimizer** steps, not micro-batches.
    With ``grad_acc > 1``, there are fewer optimizer steps than dataloader batches.

    If ``save_dir`` and ``tokenizer`` are set, checkpoints are written at epoch
    boundaries as ``{save_dir}/ep{E}/seed_{seed}``. When ``save_epochs`` is None,
    every epoch is saved; otherwise only the listed 1-based epoch numbers are
    saved (e.g. [5, 10]).

    ``bf16`` runs the forward pass under ``torch.autocast`` in bfloat16 (model
    weights and the optimizer stay in fp32). This is required when using a
    Flash Attention implementation, which only accepts fp16/bf16 tensors.
    """
    if num_epochs <= 0:
        logger.info(f"[{phase_name}] num_epochs={num_epochs}; skipping training.")
        return

    grad_acc = max(int(grad_acc), 1)
    batches_per_epoch = len(dataloader)
    opt_steps_epoch = optimizer_steps_per_epoch(batches_per_epoch, grad_acc)
    total_steps = num_epochs * opt_steps_epoch
    if total_steps <= 0:
        logger.info(f"[{phase_name}] total_steps={total_steps}; skipping training.")
        return

    logger.info(f"[{phase_name}] Moving model to device: {device}")
    model.to(device)
    model.train()
    ddp = dist.is_available() and dist.is_initialized()
    train_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if ddp else model

    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.1,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 100, 1),
        num_training_steps=total_steps,
    )

    global_step = 0
    running_loss = torch.zeros((), device=device)  # on-GPU: no host sync per micro-batch
    save_set = set(save_epochs) if save_epochs is not None else None

    logger.info(
        f"[{phase_name}] Starting training: {num_epochs} epochs, "
        f"{batches_per_epoch} batches/epoch, grad_acc={grad_acc} -> "
        f"{opt_steps_epoch} optimizer steps/epoch, "
        f"{total_steps} optimizer steps total, lr={lr:.2e}."
    )
    if save_dir is not None:
        layout_hint = os.path.join(save_dir, "ep{E}", f"seed_{seed}")
        if save_set is None:
            logger.info(
                f"[{phase_name}] Will save a checkpoint after every epoch as {layout_hint}"
            )
        else:
            logger.info(
                f"[{phase_name}] Will save checkpoints after epochs {sorted(save_set)} as {layout_hint}"
            )

    def _optimizer_update():
        nonlocal global_step, running_loss
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        if global_step % logging_steps == 0:
            avg_loss = running_loss.item() / logging_steps
            if not math.isfinite(avg_loss):
                raise RuntimeError(
                    f"[{phase_name}] Non-finite loss in the last {logging_steps} optimizer "
                    f"steps (at step {global_step}): {avg_loss!r}. This is usually an empty "
                    "tokenizer (vocab_size=0) so every token is padding and all labels are -100."
                )
            logger.info(
                f"[{phase_name}] step {global_step}/{total_steps} "
                f"- loss: {avg_loss:.4f} "
                f"- lr: {scheduler.get_last_lr()[0]:.2e}"
            )
            running_loss.zero_()

    for epoch in range(num_epochs):
        logger.info(f"[{phase_name}] Epoch {epoch + 1}/{num_epochs} started.")
        micro_in_acc = 0
        if hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)
            # With DDP, only all-reduce gradients on the last micro-batch of a window.
            sync_now = micro_in_acc + 1 == grad_acc or step == batches_per_epoch - 1
            with contextlib.nullcontext() if (not ddp or sync_now) else train_model.no_sync():
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=bf16
                ):
                    outputs = train_model(**batch)
                    loss = outputs.loss / grad_acc
                loss.backward()
            running_loss += loss.detach()
            micro_in_acc += 1

            if micro_in_acc == grad_acc:
                _optimizer_update()
                micro_in_acc = 0

        # Flush leftover micro-batches that did not fill a full grad_acc window.
        if micro_in_acc > 0:
            _optimizer_update()

        # End of epoch: optionally save a checkpoint
        epoch_num = epoch + 1
        should_save = (
            save_dir is not None
            and tokenizer is not None
            and (save_set is None or epoch_num in save_set)
        )
        if should_save:
            epoch_dir = os.path.join(save_dir, f"ep{epoch_num}", f"seed_{seed}")
            if is_main_process():
                os.makedirs(epoch_dir, exist_ok=True)
                logger.info(f"[{phase_name}] Saving epoch {epoch_num} checkpoint to {epoch_dir}")
                model.save_pretrained(epoch_dir)
            if ddp:
                dist.barrier()

    logger.info(
        f"[{phase_name}] Finished {num_epochs} epochs "
        f"({global_step}/{total_steps} optimizer steps)."
    )



def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-phase decoder-only pretraining for morphology experiments. "
        "Use --resume_from to skip Phase 1 and continue from a saved checkpoint."
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Directory of a Hugging Face saved model (e.g. a Phase-1 checkpoint). "
        "When set, Phase 1 is skipped and only Phase 2 is trained; "
        "architecture flags are ignored (loaded from checkpoint). "
        "The tokenizer is always loaded from --tokenizer_path, not the checkpoint.",
    )

    parser.add_argument(
        "--phase1_path",
        type=str,
        default=None,
        help="Text file for Phase 1 pretraining (required unless --resume_from is set).",
    )
    parser.add_argument(
        "--phase2_path",
        type=str,
        default=None,
        help="Text file for Phase 2 pretraining (e.g., German). "
        "Required unless --epochs_phase2 is 0.",
    )

    parser.add_argument(
        "--l1",
        type=str,
        required=True,
        help="Identifier for Phase 1 language (e.g., 'tur' or 'eng').",
    )
    parser.add_argument(
        "--l2",
        type=str,
        default="deu",
        help="Identifier for Phase 2 language (default: 'deu').",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="../tokenizer/tokenizer_char",
        help="Path to the character tokenizer. Always used, including with "
        "--resume_from; epoch checkpoints store model weights only.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../models",
        help="Base output directory. Checkpoints are saved as "
        "{out}/ep{E}/seed_{seed}/.",
    )

    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-GPU micro-batch size. Effective batch = batch_size * grad_acc * num_gpus.",
    )
    parser.add_argument(
        "--epochs_phase1",
        type=int,
        default=10,
        help="Number of Phase-1 epochs (default: 5).",
    )
    parser.add_argument(
        "--epochs_phase2",
        type=int,
        default=5,
        help="Number of Phase-2 epochs (default: 5). Set 0 to skip Phase 2.",
    )
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for both phases (default: 1e-4).",
    )
    parser.add_argument(
        "--save_epochs_phase1",
        type=str,
        default="5,10",
        help="Comma-separated 1-based Phase-1 epochs to checkpoint "
        "(default: '5,10'). Empty string saves every epoch.",
    )
    parser.add_argument(
        "--save_epochs_phase2",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated 1-based Phase-2 epochs to checkpoint "
        "(default: '1,2,3,4,5'). Empty string saves every epoch.",
    )
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument(
        "--length_bucketing",
        action="store_true",
        help="Group similar-length lines into the same batch (less padding, faster). "
        "Batches are still shuffled each epoch.",
    )
    parser.add_argument(
        "--token_cache_dir",
        type=str,
        default=None,
        help="Where to cache the pre-tokenized data (default: <data dir>/token_cache).",
    )
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument(
        "--reset_embeddings",
        action="store_true",
        help="Re-initialise the word-token embedding layer before Phase 2 "
        "to remove lexical knowledge acquired during Phase 1.",
    )

    parser.add_argument(
        "--save_epoch0_after_reset",
        action="store_true",
        help="With --reset_embeddings: save the weights right after the reset "
        "(before any Phase-2 training) as {out}/ep0/seed_{seed}.",
    )

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--max_position_embeddings", type=int, default=512)

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        help="Attention backend. 'flash_attention_3' needs a Hopper+ GPU "
        "(sm_90, e.g. H100/H200) and either the flash_attn_3 package or the "
        "'kernels' package (auto-fetches a prebuilt FA3 kernel from the HF Hub). "
        "Requires --bf16. Default: transformers picks 'sdpa' automatically.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Run the forward pass under bfloat16 autocast (weights/optimizer "
        "stay fp32). Required when --attn_implementation is a flash attention "
        "variant, since those kernels only accept fp16/bf16 tensors.",
    )

    args = parser.parse_args()
    if args.resume_from is None and args.phase1_path is None:
        parser.error(
            "Either --phase1_path (two-phase / Phase-1 run) or --resume_from (Phase 2 only) is required."
        )
    if args.epochs_phase2 > 0 and not args.phase2_path:
        parser.error("--phase2_path is required when --epochs_phase2 > 0.")
    if (
        args.attn_implementation
        and args.attn_implementation.startswith("flash_attention")
        and not args.bf16
    ):
        parser.error(
            f"--attn_implementation {args.attn_implementation} requires --bf16 "
            "(flash attention kernels only accept fp16/bf16 tensors, and this "
            "script does not support fp16)."
        )
    return args


def main():
    args = parse_args()
    logger.info(f"[Init] Parsed arguments: {args}")
    set_seed(args.seed)

    rank, world_size, device = setup_distributed()
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    logger.info(f"[Init] Using device: {device} (world_size={world_size})")

    save_epochs_p1 = _parse_epoch_list(args.save_epochs_phase1)
    save_epochs_p2 = _parse_epoch_list(args.save_epochs_phase2)

    if args.resume_from:
        model, tokenizer = load_model_and_tokenizer_from_checkpoint(
            args.resume_from,
            tokenizer_path=args.tokenizer_path,
            attn_implementation=args.attn_implementation,
        )
    else:
        model, tokenizer = build_model_and_tokenizer(
            tokenizer_path=args.tokenizer_path,
            lower=args.lower,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_position_embeddings=args.max_position_embeddings,
            attn_implementation=args.attn_implementation,
        )

    os.makedirs(args.out, exist_ok=True)

    if not args.resume_from:
        if args.epochs_phase1 <= 0:
            logger.info("[Phase1] epochs_phase1 <= 0; skipping Phase 1.")
            if is_main_process():
                model.save_pretrained(args.out)
        else:
            # Phase 1: pretraining on L1 (e.g., Turkish or English)
            logger.info(
                f"[Phase1] Building dataloader from {args.phase1_path} for language {args.l1}"
            )
            train_loader_p1, batches_per_epoch_p1 = create_dataloader(
                file_path=args.phase1_path,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                num_workers=args.cpus,
                seed=args.seed,
                length_bucketing=args.length_bucketing,
                cache_dir=args.token_cache_dir,
            )
            opt_steps_epoch_p1 = optimizer_steps_per_epoch(
                batches_per_epoch_p1, args.grad_acc
            )
            total_steps_p1 = args.epochs_phase1 * opt_steps_epoch_p1
            logger.info(
                f"[Phase1] Training on {args.l1}: {args.epochs_phase1} epochs "
                f"x {batches_per_epoch_p1} batches/epoch, grad_acc={args.grad_acc} "
                f"-> {opt_steps_epoch_p1} optimizer steps/epoch "
                f"= {total_steps_p1} optimizer steps total "
                f"(lr={args.lr:.2e}, seed={args.seed})"
            )
            train_one_phase(
                model=model,
                dataloader=train_loader_p1,
                num_epochs=args.epochs_phase1,
                lr=args.lr,
                grad_acc=args.grad_acc,
                logging_steps=args.logging_steps,
                device=device,
                phase_name="Phase1",
                save_dir=args.out,
                tokenizer=tokenizer,
                save_epochs=save_epochs_p1,
                seed=args.seed,
                bf16=args.bf16,
            )

            # Convenience alias for the final Phase-1 state
            # phase1_dir = os.path.join(args.out, f"phase1_{args.l1}")
            # os.makedirs(phase1_dir, exist_ok=True)
            # logger.info(f"[Phase1] Saving final Phase-1 checkpoint to {phase1_dir}")
            # model.save_pretrained(phase1_dir)
            # tokenizer.save_pretrained(phase1_dir)
    else:
        logger.info(
            "[Init] Skipping Phase 1 (--resume_from set); training Phase 2 only."
        )

    if args.epochs_phase2 <= 0:
        logger.info("[Phase2] epochs_phase2 <= 0; skipping Phase 2.")
        return

    # Optionally reset embedding weights before Phase 2
    if args.reset_embeddings:
        logger.info(
            "[Phase2] --reset_embeddings is set: re-initialising embedding layer "
            "to remove lexical knowledge from Phase 1."
        )
        reset_embedding_weights(model)
        if args.save_epoch0_after_reset:
            ep0_dir = os.path.join(args.out, "ep0", f"seed_{args.seed}")
            if is_main_process():
                os.makedirs(ep0_dir, exist_ok=True)
                logger.info(f"[ResetEmb] Saving epoch-0 (post-reset) weights to {ep0_dir}")
                model.save_pretrained(ep0_dir)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    # Phase 2: continue pretraining on L2 (typically German)
    logger.info(
        f"[Phase2] Building dataloader from {args.phase2_path} for language {args.l2}"
    )
    train_loader_p2, batches_per_epoch_p2 = create_dataloader(
        file_path=args.phase2_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.cpus,
        seed=args.seed,
        length_bucketing=args.length_bucketing,
        cache_dir=args.token_cache_dir,
    )
    opt_steps_epoch_p2 = optimizer_steps_per_epoch(
        batches_per_epoch_p2, args.grad_acc
    )
    total_steps_p2 = args.epochs_phase2 * opt_steps_epoch_p2
    logger.info(
        f"[Phase2] Training on {args.l2}: {args.epochs_phase2} epochs "
        f"x {batches_per_epoch_p2} batches/epoch, grad_acc={args.grad_acc} "
        f"-> {opt_steps_epoch_p2} optimizer steps/epoch "
        f"= {total_steps_p2} optimizer steps total "
        f"(lr={args.lr:.2e}, seed={args.seed})"
    )
    train_one_phase(
        model=model,
        dataloader=train_loader_p2,
        num_epochs=args.epochs_phase2,
        lr=args.lr,
        grad_acc=args.grad_acc,
        logging_steps=args.logging_steps,
        device=device,
        phase_name="Phase2",
        save_dir=args.out,
        tokenizer=tokenizer,
        save_epochs=save_epochs_p2,
        seed=args.seed,
        bf16=args.bf16,
    )

    # Final checkpoint
    # final_dir = os.path.join(args.out, f"phase2_{args.l1}_{args.l2}")
    # os.makedirs(final_dir, exist_ok=True)
    # logger.info(f"[Final] Saving final checkpoint to {final_dir}")
    # model.save_pretrained(final_dir)
    # tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
