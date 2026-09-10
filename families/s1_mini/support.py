"""Family-owned model and task support for s1_mini (S1-mini by Superwhisper)."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    required_files=("banner.jpg",),
    tasks=("text_generation",),
    default_task="text_generation",
)
