"""Montessori-Instruct Unified Modular Package (`montessori`).

Exposes reusable components for both Text (Domain CPT/QA) and Multimodal (Datawheel VLM)
Student-Aware Data Synthesis & API Preference Alignment:

  - Stage 1 (`probing_generate`): `run_probing_generation`, `verify_and_enrich_item`
  - Stage 2 (`influence_scorer`): `TorchStudentInfluenceEngine`, `run_influence_scoring`
  - Stage 3 (`build_dpo_pairs`): `build_dpo_and_sft_datasets`
  - Stage 4 (`api_preference_synthesizer`):
      * `PreferenceMemoryBank` (In-Context Contrastive Steering & Meta-Prompt Evolution)
      * `compute_composite_preference_score` (Influence + Verification + Calibration)
      * `export_student_multi_preference_datasets` (Student DPO / KTO / GRPO)
      * `run_api_preference_synthesizer`
"""
from __future__ import annotations

import sys
from pathlib import Path

_MONTESSORI_ROOT = Path(__file__).resolve().parent
if str(_MONTESSORI_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONTESSORI_ROOT))

from api_preference_synthesizer import (  # noqa: E402
    API_SYNTHESIS_PERSONAS,
    PreferenceMemoryBank,
    compute_composite_preference_score,
    diagnose_negative_defect,
    export_student_multi_preference_datasets,
    run_api_preference_synthesizer,
)
from build_dpo_pairs import (  # noqa: E402
    build_dpo_and_sft_datasets,
    normalize_teacher_prompt,
)
from influence_scorer import (  # noqa: E402
    TorchStudentInfluenceEngine,
    compute_mock_influence,
    encode_assistant_supervised_example,
    run_influence_scoring,
)
from probing_generate import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    TASK_TYPES,
    build_student_sft_messages,
    format_source_bundle,
    run_probing_generation,
    verify_and_enrich_item,
)

__all__ = [
    "API_SYNTHESIS_PERSONAS",
    "DEFAULT_SYSTEM_PROMPT",
    "PreferenceMemoryBank",
    "TASK_TYPES",
    "TorchStudentInfluenceEngine",
    "build_dpo_and_sft_datasets",
    "build_student_sft_messages",
    "compute_composite_preference_score",
    "compute_mock_influence",
    "diagnose_negative_defect",
    "encode_assistant_supervised_example",
    "export_student_multi_preference_datasets",
    "format_source_bundle",
    "normalize_teacher_prompt",
    "run_api_preference_synthesizer",
    "run_influence_scoring",
    "run_probing_generation",
    "verify_and_enrich_item",
]
