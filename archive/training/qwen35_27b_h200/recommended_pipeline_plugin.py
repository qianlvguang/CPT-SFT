"""固定推荐配方：累计同一个 LoRA；S2/S3 先加载视觉 adapter 再冻结。"""
import os
from swift.dataset.preprocessor.core import RowPreprocessor

_original_preprocess = RowPreprocessor.batched_preprocess


def _preprocess(self, batched_row, *, strict, ignore_max_length_error):
    row = dict(batched_row)
    if "channel" not in row and "variant" in row:
        row["channel"] = [f"V{int(v)}" if v is not None else None for v in row["variant"]]
    # 不允许框架忽略 MaxLengthError 并悄悄丢行，也不截断回答来凑长度。
    result = _original_preprocess(self, row, strict=True,
                                  ignore_max_length_error=False)
    if "variant" in row and len(result.get("messages", [])) != len(row["variant"]):
        raise ValueError("推荐路线不允许预处理静默丢行；请检查超长或异常样本。")
    return result


if not getattr(RowPreprocessor.batched_preprocess, "_recommended_patched", False):
    _preprocess._recommended_patched = True
    RowPreprocessor.batched_preprocess = _preprocess
    print("[recommended_plugin] variant->channel patch installed")

from swift.megatron.utils import utils
_original_config = utils.LoraConfig

def _lora_config(*args, **kwargs):
    # 三阶段都重建相同模块和形状，保留冻结的视觉 LoRA 供前向与保存。
    # merger 先匹配，避免通用 visual 模式把 merger 也降为 r16。
    kwargs["rank_pattern"] = {r"visual\.merger\..*": 32, r"visual\..*": 16}
    kwargs["alpha_pattern"] = {r"visual\.merger\..*": 64, r"visual\..*": 32}
    config = _original_config(*args, **kwargs)
    print(f"[recommended_plugin] rank_pattern={config.rank_pattern} alpha_pattern={config.alpha_pattern}")
    return config

if not getattr(utils.LoraConfig, "_recommended_patched", False):
    _lora_config._recommended_patched = True
    utils.LoraConfig = _lora_config


def enforce_trainable_adapters(models, stage):
    """加载完成后、optimizer 创建前执行；冻结不是移除 adapter。"""
    visual_lora = llm_lora = trainable = 0
    for model in models:
        for name, parameter in model.named_parameters():
            visual = name.startswith("visual.") or ".visual." in name
            lora = ".lora_A." in name or ".lora_B." in name
            if lora:
                if visual:
                    visual_lora += parameter.numel()
                else:
                    llm_lora += parameter.numel()
            parameter.requires_grad_(lora and (stage == 1 or not visual))
            if parameter.requires_grad:
                trainable += parameter.numel()
        if stage > 1:
            # 冻结视觉 LoRA 的 dropout 也应关闭，避免随机视觉扰动。
            for name, module in model.named_modules():
                if name.startswith("visual.") or ".visual." in name:
                    dropout = getattr(module, "lora_dropout", None)
                    if dropout is not None:
                        for layer in dropout.values():
                            if hasattr(layer, "p"):
                                layer.p = 0.0
    if visual_lora == 0 or llm_lora == 0 or trainable == 0:
        raise RuntimeError("没有同时发现视觉和 LLM LoRA；拒绝丢失视觉适配或启动空训练。")
    print(f"[recommended_plugin] stage={stage} visual_lora={visual_lora} "
          f"llm_lora={llm_lora} trainable={trainable} visual_frozen={stage > 1}")
    return {"visual_lora": visual_lora, "llm_lora": llm_lora, "trainable": trainable}


# 4.3.2 的顺序：prepare_model -> _prepare_peft_model（含 adapters 加载）
# -> wrap_model -> get_optimizer_and_scheduler。此处冻结才既保留视觉增量，
# 又不会给视觉参数创建梯度和 optimizer state。
from swift.megatron.trainers.base import BaseMegatronTrainer
_original_prepare = BaseMegatronTrainer._prepare_peft_model


def _prepare(self, models):
    stage = int(os.environ["PIPELINE_STAGE"])
    if stage > 1 and len(self.args.adapters) != 1:
        raise RuntimeError("S2/S3 必须加载上一阶段的一份累计 adapter")
    peft_models = _original_prepare(self, models)
    enforce_trainable_adapters(models, stage)
    return peft_models


if not getattr(BaseMegatronTrainer._prepare_peft_model, "_recommended_patched", False):
    _prepare._recommended_patched = True
    BaseMegatronTrainer._prepare_peft_model = _prepare
