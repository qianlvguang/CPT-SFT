#!/usr/bin/env python3
"""Qwen3.5-27B：唯一推荐的 V1～V4 三阶段训练路线。

plan 仅展示方案；preflight 校验远端数据；smoke/run 真正启动训练。
所有阶段必须完整完成至少一遍数据，不自动启动历史 sweep。
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time

RECIPE = {
    "name": "v1_v4_s1l_s2_s3_cumulative_adapter_v2",
    "seed": 42, "gbs": 32, "mbs": 2, "max_length": 8192,
    "tp": 4, "pp": 1, "cp": 1, "ep": 1, "gpus": 8,
    "llm_rank": 64, "llm_alpha": 128, "dropout": 0.05,
    "warmup": 0.03, "weight_decay": 0.01, "clip": 1.0,
    "image_tokens": 16384, "video_tokens": 128, "fps_max_frames": 16,
    "replay_rows": 120_000,
    "stages": [
        {"stage": 1, "pool": "mini", "epochs": 2, "lr": 3e-5, "min_lr": 3e-6,
         "visual": "lora", "vit_rank": 16, "vit_alpha": 32, "vit_lr": 1e-5,
         "merger_rank": 32, "merger_alpha": 64, "aligner_lr": 1e-5},
        {"stage": 2, "pool": "medium", "epochs": 2, "lr": 2e-5, "min_lr": 2e-6,
         "visual": "frozen_adapter"},
        {"stage": 3, "pool": "final", "epochs": 1, "lr": 1e-5, "min_lr": 1e-6,
         "visual": "frozen_adapter"},
    ],
}
EXPECTED = {"mini": [360000, 200000, 120000, 120000],
            "medium": [120000, 90000, 150000, 180000],
            "final": [40113, 230100, 423370, 403947]}
# 仅用于存档间隔和空间预算；不用于限制真实训练步数。
ROWS_PER_WINDOW_ESTIMATE = 10.2
SOURCE = Path(__file__).resolve()


def say(message):
    print(message, flush=True)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        temp = f.name
    os.replace(temp, path)


def check_safetensors(path):
    """检查头部和数据区长度；不加载 27B 张量到内存。"""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"权重文件截断：{path}")
        n = struct.unpack("<Q", raw)[0]
        if n < 2 or n > min(100_000_000, size - 8):
            raise ValueError(f"非法 safetensors 头：{path}")
        header = json.loads(f.read(n))
    intervals = []
    names = set()
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
              "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
              "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        shape = entry["shape"]
        if (not all(isinstance(x, int) and x >= 0 for x in [start, end, *shape])
                or start > end or end > size - 8 - n):
            raise ValueError(f"权重数据区截断/偏移异常：{path}:{name}")
        dtype = entry["dtype"]
        if dtype not in widths or math.prod(shape) * widths[dtype] != end - start:
            raise ValueError(f"权重 dtype/shape/长度异常：{path}:{name}")
        names.add(name)
        intervals.append((start, end))
    if not names:
        raise ValueError(f"空权重：{path}")
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"权重数据区不连续：{path}")
        cursor = end
    if cursor != size - 8 - n:
        raise ValueError(f"权重数据区长度不一致：{path}")
    return names


def model_fingerprint(model):
    model = Path(model)
    config = model / "config.json"
    if not config.is_file():
        raise ValueError(f"缺少模型配置：{config}")
    json.loads(config.read_text())
    index = model / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        if not mapping:
            raise ValueError(f"空权重索引：{index}")
        shards = sorted(set(mapping.values()))
    else:
        mapping = None
        shards = ["model.safetensors"]
    for name in shards:
        if Path(name).name != name:
            raise ValueError(f"非法权重索引路径：{name}")
        tensors = check_safetensors(model / name)
        if mapping and not {k for k, v in mapping.items() if v == name} <= tensors:
            raise ValueError(f"权重与索引不一致：{model / name}")
    # 完整读取权重内容；不会以同路径/同大小替换的模型冒充旧父档。
    files = sorted(set(shards) | {x.name for x in model.iterdir()
                   if x.is_file() and (x.suffix in {".json", ".safetensors", ".model", ".txt"}
                                       or x.name.endswith(".jinja"))})
    return {name: sha256(model / name) for name in files}


def adapter_fingerprint(adapter):
    """只检查增量权重；所有阶段都必须保留 S1 的视觉 LoRA。"""
    adapter = Path(adapter)
    config = adapter / "adapter_config.json"
    if not config.is_file():
        raise ValueError(f"缺少 adapter 配置：{config}")
    info = json.loads(config.read_text())
    if info.get("peft_type") != "LORA" or info.get("r") != 64 or info.get("lora_alpha") != 128:
        raise ValueError(f"adapter 必须为本路线的 LORA r64/a128：{config}")
    if info.get("modules_to_save") or info.get("bias", "none") != "none":
        raise ValueError(f"不允许额外全参模块或 bias：{config}")
    if info.get("rank_pattern") != {r"visual\.merger\..*": 32, r"visual\..*": 16}:
        raise ValueError(f"adapter 的视觉异构 rank_pattern 与本路线不一致：{config}")
    if info.get("alpha_pattern") != {r"visual\.merger\..*": 64, r"visual\..*": 32}:
        raise ValueError(f"adapter 的视觉异构 alpha_pattern 与本路线不一致：{config}")
    weights = adapter / "adapter_model.safetensors"
    names = check_safetensors(weights)
    if any(".lora_A." not in name and ".lora_B." not in name for name in names):
        raise ValueError(f"adapter 含非 LoRA 张量：{weights}")
    a_names = {name for name in names if ".lora_A." in name}
    b_names = {name for name in names if ".lora_B." in name}
    if not a_names or {name.replace(".lora_A.", ".lora_B.") for name in a_names} != b_names:
        raise ValueError(f"adapter 的 A/B 矩阵不完整：{weights}")
    if not any("visual." in name for name in names) or not any("visual." not in name for name in names):
        raise ValueError(f"adapter 必须同时保留视觉 LoRA 和 LLM LoRA：{weights}")
    files = [config, weights]
    args = adapter / "args.json"
    if args.is_file():
        files.append(args)
    result = {path.name: sha256(path) for path in files}
    # 单独记录视觉增量，供下一阶段验证冻结权重确实未改变。
    visual_digest = hashlib.sha256()
    with weights.open("rb") as f:
        header_length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_length))
        for name in sorted(x for x in names if "visual." in x):
            entry = header[name]
            visual_digest.update(canonical([name, entry["shape"], entry["dtype"]]).encode())
            start, end = entry["data_offsets"]
            f.seek(8 + header_length + start)
            remaining = end - start
            while remaining:
                block = f.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"视觉权重读取时被截断：{weights}")
                visual_digest.update(block)
                remaining -= len(block)
    result["__visual_lora_sha256__"] = visual_digest.hexdigest()
    return result


def paths():
    work = Path(os.environ.get("WORK_DIR", "/data/autovision-cfs/autovision-notebook")).expanduser()
    data = Path(os.environ.get("TEST_DIR", str(work / "algo4/tpz/test"))).expanduser().resolve()
    base = Path(os.environ.get("BASE_MODEL", str(work / "algo4/models/Qwen3.5-27B"))).expanduser().resolve()
    root = Path(os.environ.get("OUTPUT_ROOT", str(work / "algo4/tpz/ft/s1_s3_recommended"))).expanduser().resolve()
    return work, data, base, root


def data_paths(data, stage):
    return [data / f"train_caption_ultra_x_v{v}{stage['pool']}.parquet" for v in range(1, 5)]


def preflight(data, root):
    """按 variant + 行级键检查同类数据分区，不按 (task,sid) 丢图。"""
    import pyarrow.parquet as pq
    root.mkdir(parents=True, exist_ok=True)
    report = []
    with tempfile.TemporaryDirectory(prefix="preflight_", dir=root) as tmp:
        db = sqlite3.connect(str(Path(tmp) / "keys.sqlite"))
        try:
            db.execute("CREATE TABLE seen (variant INTEGER, key TEXT, pool TEXT, PRIMARY KEY(variant,key))")
            for stage in RECIPE["stages"]:
                for variant, path in enumerate(data_paths(data, stage), 1):
                    say(f"校验数据：{path}")
                    pf = pq.ParquetFile(path)
                    required = {"messages", "images", "sample_id", "cate_id", "variant"}
                    missing = required - set(pf.schema_arrow.names)
                    if missing or pf.metadata.num_rows <= 0:
                        raise ValueError(f"空数据或缺列 {sorted(missing)}：{path}")
                    expected = EXPECTED[stage["pool"]][variant - 1]
                    if pf.metadata.num_rows != expected:
                        say(f"提示：{path.name} 实际 {pf.metadata.num_rows:,} 行，历史记录 {expected:,} 行；使用实际完整数据。")
                    cols = ["sample_id", "cate_id", "images", "variant", "messages"]
                    check_thinking = variant == 4 and stage["pool"] == "final"
                    if check_thinking:
                        if "enable_thinking" not in pf.schema_arrow.names:
                            raise ValueError(f"V4 final 缺少 enable_thinking：{path}")
                        cols.append("enable_thinking")
                    for batch in pf.iter_batches(batch_size=2048, columns=cols):
                        records = []
                        for row in batch.to_pylist():
                            if row["variant"] != variant or row["sample_id"] is None or row["cate_id"] is None:
                                raise ValueError(f"variant/行级键异常：{path}")
                            if not row["messages"]:
                                raise ValueError(f"空 messages：{path}")
                            if check_thinking and row["enable_thinking"] is not False:
                                raise ValueError(f"V4 final 的 enable_thinking 必须为 false：{path}")
                            key = canonical([row["sample_id"], row["cate_id"], row["images"]])
                            records.append((variant, key, stage["pool"]))
                        try:
                            db.executemany("INSERT INTO seen VALUES (?,?,?)", records)
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"同一 V{variant} 内存在重复行或 mini/medium/final 交集：{path}；不自动去重。") from exc
                        db.commit()
                    report.append({"stage": stage["stage"], "variant": variant, "path": str(path),
                                   "rows": pf.metadata.num_rows, "sha256": sha256(path)})
        finally:
            db.close()
    atomic_json(root / "data_manifest.json", report)
    return report


def make_replay(data, root, report):
    """V1 mini+medium 中按固定随机种子无放回抽 120k，逐批写出。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    sources = [x for x in report if x["variant"] == 1 and x["stage"] in (1, 2)]
    total = sum(x["rows"] for x in sources)
    count = min(RECIPE["replay_rows"], total)
    signature = {"sources": sources, "seed": RECIPE["seed"], "rows": count,
                 "algorithm": "global_random_sample_without_replacement_v1"}
    out = root / "data/v1_replay.parquet"
    marker = root / "data/v1_replay.json"
    if marker.is_file() and out.is_file():
        prior = json.loads(marker.read_text())
        if prior.get("signature") == signature and prior.get("sha256") == sha256(out):
            return {"path": str(out), "rows": count, "sha256": prior["sha256"]}
    picks = sorted(random.Random(RECIPE["seed"]).sample(range(total), count))
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".parquet.partial")
    writer = None
    offset = 0
    written = 0
    try:
        for source in sources:
            for batch in pq.ParquetFile(source["path"]).iter_batches(batch_size=4096):
                left = bisect.bisect_left(picks, offset)
                right = bisect.bisect_left(picks, offset + batch.num_rows)
                if right > left:
                    chosen = pa.Table.from_batches([batch]).take(pa.array(
                        [i - offset for i in picks[left:right]], type=pa.int64()))
                    if writer is None:
                        writer = pq.ParquetWriter(temp, chosen.schema, compression="zstd")
                    # 两个源若 schema 不兼容，直接报错，保留原始数据。
                    writer.write_table(chosen)
                    written += chosen.num_rows
                offset += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if written != count:
        raise ValueError(f"回放样本数异常：{written} != {count}")
    os.replace(temp, out)
    digest = sha256(out)
    atomic_json(marker, {"signature": signature, "sha256": digest})
    return {"path": str(out), "rows": count, "sha256": digest}


def runtime_versions():
    result = {}
    for name in ["ms-swift", "megatron-core", "mcore-bridge", "peft", "torch", "transformers", "pyarrow"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"训练环境缺少 {name}；请使用既有远端训练环境。")
    if result["ms-swift"] != "4.3.2" or result["megatron-core"] != "0.18.0":
        raise RuntimeError(f"该配方按历史环境验证；版本不一致，请先校对参数/API：{result}")
    return result


def stage_command(stage, model, datasets, out, plugin, smoke, save_steps, adapter=None):
    cmd = ["megatron", "sft"]
    def add(key, value):
        cmd.extend([f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)])
    add("model", model)
    if adapter is not None:
        add("adapters", adapter)
    cmd.extend(["--dataset", *[x["path"] for x in datasets]])
    params = {"split_dataset_ratio": 0, "load_from_cache_file": True, "save_safetensors": True,
              "strict": True, "truncation_strategy": "delete", "dataset_shuffle": True, "data_seed": 42,
              "tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 1,
              "context_parallel_size": 1, "expert_model_parallel_size": 1,
              "sequence_parallel": True, "use_distributed_optimizer": False,
              "recompute_granularity": "selective", "attention_backend": "flash",
              "packing": True, "padding_free": True, "tuner_type": "lora",
              "target_modules": "all-linear", "lora_rank": 64, "lora_alpha": 128,
              "lora_dropout": 0.05, "merge_lora": False, "freeze_llm": False,
              # 先重建视觉 LoRA 并加载旧权重；插件在 optimizer 创建之前冻结。
              # 直接 freeze_vit=true 会阻止建视觉 LoRA，丢失 S1 视觉适配。
              "freeze_vit": False, "freeze_aligner": False,
              "vit_gradient_checkpointing": stage["stage"] == 1,
              "external_plugins": plugin, "enable_channel_loss": True,
              "micro_batch_size": 2, "global_batch_size": 32, "max_length": 8192,
              "lr": stage["lr"], "min_lr": stage["min_lr"], "lr_warmup_fraction": 0.03,
              "lr_decay_style": "cosine", "weight_decay": 0.01, "clip_grad": 1.0, "seed": 42,
              "cross_entropy_loss_fusion": True, "finetune": True,
              "no_save_optim": True, "no_save_rng": True,
              "save_strategy": "steps", "save_steps": save_steps,
              # 禁止轮替删除候选；disk 预检失败则停，不自行清理。
              "save_total_limit": 1000000, "add_version": False, "output_dir": out,
              "dataloader_num_workers": 32, "dataset_num_proc": 16, "packing_num_proc": 16}
    params["train_iters" if smoke else "num_train_epochs"] = 10 if smoke else stage["epochs"]
    if stage["visual"] == "lora":
        params.update(vit_lr=stage["vit_lr"], aligner_lr=stage["aligner_lr"])
    for key, value in params.items():
        add(key, value)
    return cmd


def run_logged(cmd, env, cwd, log):
    """subprocess 退出码直接传递；异常/中断时回收整个训练进程组。"""
    import signal
    with Path(log).open("w", encoding="utf-8") as output:
        process = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    say(f"训练运行中；日志：{log}")
            if code:
                raise RuntimeError(f"训练失败，退出码 {code}；查看 {log}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def candidates(out):
    found = []
    for path in out.glob("checkpoint-*"):
        try:
            step = int(path.name[len("checkpoint-"):])
        except ValueError:
            continue
        if path.is_dir():
            found.append((step, path))
    if not found:
        raise RuntimeError(f"训练成功退出但没有 adapter checkpoint：{out}")
    return sorted(found)


def select_checkpoint(out, stage, evaluator, smoke, env, work):
    choices = candidates(out)
    last_step = choices[-1][0]
    latest = out / "latest_checkpointed_iteration.txt"
    if latest.is_file() and latest.read_text().strip().isdigit():
        if int(latest.read_text().strip()) != last_step:
            raise RuntimeError("最终 checkpoint 指针与 adapter 档不一致")
    # 完整训练终点本身必须可用；不能以早期档掩盖最终落盘失败。
    last_fingerprint = adapter_fingerprint(choices[-1][1])
    if smoke or not evaluator:
        return choices[-1][1], {"mode": "smoke" if smoke else "final_unscored",
                               "step": last_step}, last_fingerprint
    # 由实际终点步数确定至少一遍数据的门槛；S3 一 epoch 只选终点。
    minimum = math.ceil(last_step / stage["epochs"])
    scores = []
    for step, model in choices:
        if step < minimum:
            continue
        fingerprint = last_fingerprint if step == last_step else adapter_fingerprint(model)
        metrics_path = out / f"selection_{step}.json"
        # 固定接口：bash EVAL_SCRIPT ADAPTER STAGE METRICS_JSON，底座在环境中。
        subprocess.run(["bash", str(evaluator), str(model), str(stage["stage"]),
                        str(metrics_path)], env=env, cwd=work, check=True)
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("dataset_id") != "eval_v3_5658":
            raise ValueError("评测必须固定 dataset_id=eval_v3_5658；禁止混用全量口径选档")
        for key in ["accuracy", "f1"]:
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"评测字段 {key} 必须在 [0,1] 内")
        scores.append((metrics["accuracy"], metrics["f1"], -step, model, metrics, fingerprint))
    winner = max(scores, key=lambda x: x[:3])
    atomic_json(out / "selection_report.json", [
        {"adapter": str(x[3]), "metrics": x[4]} for x in scores])
    return winner[3], {"mode": "eval_v3_5658", "metrics": winner[4]}, winner[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "preflight", "smoke", "run"], nargs="?", default="plan")
    args = parser.parse_args()
    work, data, base, root = paths()
    smoke = args.mode == "smoke"
    if smoke:
        root = root.with_name(root.name + "_smoke")
    evaluator = os.environ.get("EVAL_SCRIPT", "")
    if args.mode == "plan":
        say(canonical({"recipe": RECIPE, "data": str(data), "base": str(base), "output": str(root),
                       "data_files": {f"s{x['stage']}": [str(p) for p in data_paths(data, x)]
                                      for x in RECIPE["stages"]},
                       "selection": "eval_v3_5658" if evaluator else "final_unscored"}))
        say("独立原始样本（按历史行数）：2,437,530；S3 另加 V1 回放 120,000。")
        say("没有 EVAL_SCRIPT 时按终点衔接，不宣称选出了实测最佳档。")
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".pipeline.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"已有任务或残留锁：{lock}；核实 owner.json 所属任务已停止后再处理。")
    try:
        atomic_json(lock / "owner.json", {"pid": os.getpid(), "host": os.uname().nodename,
                                         "started": time.time(), "mode": args.mode})
        report = preflight(data, root)
        if args.mode == "preflight":
            say(f"数据预检通过，清单：{root / 'data_manifest.json'}")
            return
        versions = runtime_versions()
        if not work.is_dir() or not shutil.which("megatron"):
            raise RuntimeError("缺少工作目录或 megatron 命令")
        if evaluator:
            evaluator = Path(evaluator).expanduser().resolve()
            if not evaluator.is_file():
                raise ValueError(f"EVAL_SCRIPT 不存在：{evaluator}")
        elif not smoke:
            say("提示：未配置评测器，各阶段将使用终点，标记 final_unscored。")
        replay = make_replay(data, root, report)
        plugin = root / "pipeline_plugin.py"
        plugin_source = SOURCE.with_name("recommended_pipeline_plugin.py").read_text()
        plugin.write_text(plugin_source)
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   CUDA_DEVICE_MAX_CONNECTIONS="1", TORCH_COMPILE_DISABLE="1",
                   WANDB_MODE=os.environ.get("WANDB_MODE", "disabled"),
                   TOKENIZERS_PARALLELISM="False", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", RAYON_NUM_THREADS="1",
                   IMAGE_MAX_TOKEN_NUM="16384", VIDEO_MAX_TOKEN_NUM="128", FPS_MAX_FRAMES="16",
                   NPROC_PER_NODE="8", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
                   MODELSCOPE_CACHE_HOME=os.environ.get("MODELSCOPE_CACHE_HOME", str(work / "logos-remote/sft/cache")))
        devices = env["CUDA_VISIBLE_DEVICES"].split(",")
        if len(devices) != 8 or len(set(devices)) != 8:
            raise ValueError("本路线要求 8 张不同的 H200，TP4 × DP2")
        if "WORLD_SIZE" in env or "RANK" in env:
            raise ValueError("请在普通单节点 shell 启动，不要嵌套 torchrun 环境")
        parent = None
        say(f"校验并计算底座内容指纹：{base}（会顺序读取约 51 GB 权重）")
        base_fingerprint = model_fingerprint(base)
        parent_fingerprint = None
        env["PIPELINE_BASE_MODEL"] = str(base)
        results = {}
        eval_signature = sha256(evaluator) if evaluator else None
        for stage in RECIPE["stages"]:
            k = stage["stage"]
            datasets = [x for x in report if x["stage"] == k]
            if k == 3:
                datasets = [*datasets, replay]
            rows = sum(x["rows"] for x in datasets)
            estimated_iters = math.ceil(rows / ROWS_PER_WINDOW_ESTIMATE / RECIPE["gbs"] * stage["epochs"])
            # 大约每 1/4 epoch 一档；真实训练仍由 num_train_epochs 控制。
            save_steps = 1000000 if smoke else max(1, round(rows / ROWS_PER_WINDOW_ESTIMATE / RECIPE["gbs"] / 4))
            sig = {"recipe": RECIPE, "stage": stage, "datasets": datasets,
                   "base": base_fingerprint, "parent_adapter": parent_fingerprint,
                   "runner_sha256": sha256(SOURCE),
                   "plugin_sha256": sha256(plugin), "versions": versions,
                   "smoke": smoke, "save_steps": save_steps, "eval_script_sha256": eval_signature}
            signature = hashlib.sha256(canonical(sig).encode()).hexdigest()
            out = root / f"s{k}"
            marker = out / "complete.json"
            if marker.is_file():
                prior = json.loads(marker.read_text())
                if prior.get("signature") != signature:
                    raise RuntimeError(f"{out} 的配方/数据/父档/代码/环境已变化；请指定新的 OUTPUT_ROOT，旧结果不会覆盖。")
                chosen = Path(prior["selected_adapter"])
                if adapter_fingerprint(chosen) != prior["adapter_fingerprint"]:
                    raise RuntimeError(f"已完成权重被改变：{chosen}")
                parent, parent_fingerprint = chosen, prior["adapter_fingerprint"]
                results[f"s{k}"] = prior
                say(f"复用完整且指纹匹配的 S{k}：{chosen}")
                continue
            if out.exists() and any(out.iterdir()):
                raise RuntimeError(f"{out} 有未完成输出；不自动挪走/删除，也不跨阶段 resume。换 OUTPUT_ROOT 后重跑。")
            estimated_saves = 1 if smoke else estimated_iters // save_steps + 1
            # NFS 配额可能不同于 disk_usage；仍给出保守的常规空间检查。
            required = (estimated_saves * 3 + 10) * 1024**3
            if shutil.disk_usage(root).free < required:
                raise RuntimeError(f"S{k} 预计需 {required / 1024**3:.0f} GiB 空闲（含全部候选），可见空间不足；不自动清理。")
            out.mkdir(parents=True, exist_ok=True)
            command = stage_command(stage, base, datasets, out, plugin, smoke, save_steps, adapter=parent)
            atomic_json(out / "launch_manifest.json", {"signature": signature, "config": sig,
                                                       "command": command, "environment": {
                                                           key: env[key] for key in ["IMAGE_MAX_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM",
                                                                                   "FPS_MAX_FRAMES", "NPROC_PER_NODE", "CUDA_VISIBLE_DEVICES"]}})
            # 新签名使用独立持久缓存，避免沿用历史默默丢行后的缓存。
            cache = Path(env["MODELSCOPE_CACHE_HOME"]) / "recommended" / signature
            stage_env = dict(env, PIPELINE_STAGE=str(k), MODELSCOPE_CACHE=str(cache),
                             MODELSCOPE_CACHE_HOME=str(cache))
            say(f"S{k}：{rows:,} 行，epochs={stage['epochs']}，存档间隔={save_steps}，日志={out / 'train.log'}")
            run_logged(command, stage_env, work, out / "train.log")
            selected, selection, selected_fingerprint = select_checkpoint(out, stage, evaluator, smoke, stage_env, work)
            if parent_fingerprint is not None:
                if selected_fingerprint["__visual_lora_sha256__"] != parent_fingerprint["__visual_lora_sha256__"]:
                    raise RuntimeError(f"S{k} 的冻结视觉 LoRA 内容发生变化；拒绝衔接下一阶段。")
            parent, parent_fingerprint = selected, selected_fingerprint
            complete = {"signature": signature, "base_model": str(base), "selected_adapter": str(parent),
                        "adapter_fingerprint": parent_fingerprint, "selection": selection,
                        "finished": time.time()}
            # 只有成功退出、选档成功、完整权重校验后才发布完成标记。
            atomic_json(marker, complete)
            results[f"s{k}"] = complete
        atomic_json(root / "result.json", results)
        say(f"三阶段完成；底座：{base}；S3 累积 adapter：{parent}；记录：{root / 'result.json'}")
    finally:
        # 只释放本进程创建的锁；不删除任何模型或历史输出。
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    import signal

    def _stop(signum, frame):
        raise KeyboardInterrupt(f"收到停止信号 {signum}")

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _stop)
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        say(f"[失败] {error}")
        sys.exit(1)
