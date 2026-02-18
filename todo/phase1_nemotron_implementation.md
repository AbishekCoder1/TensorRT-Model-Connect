# Phase 1: Nemotron-4 Plugin + NVIDIA Model Validation — COMPLETED

## Status: Phase 1 DONE

All Phase 1 tasks completed:

### Completed
1. **`relu2`/`squared_relu` activation** in `graph_ops.py` (line ~313)
2. **`norm_eps` config support** in `config.py` (line ~73)
3. **Nemotron family plugin** at `trtf_build/trtf_build/families/nemotron.py` (~175 lines)
4. **6 E2E entries** in `tests/e2e/engines.json`
5. **Unit tests** for relu2/squared_relu in `tests/builder/test_graph_ops.py`
6. **Docs updated**: Home.md, Architecture-Extensibility-Assessment.md, Source-Layout.md, WORKLOG.md, CLAUDE.md, standard_decoder_builder.py docstring

### Remaining Phases (not yet implemented)

See `todo/nvidia_hf_models_plan.md` for the full 5-phase plan:
- **Phase 2**: Cosmos-Reason2-2B VL validation (~1-2 days)
- **Phase 3**: Embedding/Reranking runtime (~1-2 weeks, new C++ backend)
- **Phase 4**: Hybrid backend for Hymba + Nemotron-Flash (~3 weeks, new C++ backend)
- **Phase 5**: Encoder-Decoder for Nemotron-Parse + OCR (~2-3 weeks, new C++ backend)

### Validation (needs container)
```bash
# Tier 1: Unit tests
.venv/bin/python -m pytest tests/builder/ -v
.venv/bin/python -m pytest tests/tools/ -v
ctest --test-dir build --output-on-failure

# Tier 2: Graph-op GPU tests (verify relu2)
.venv/bin/python -m pytest tests/builder/test_graph_ops.py -v -m trt

# Tier 3: E2E smoke (Nemotron-Mini-4B)
.venv/bin/python -m pytest tests/e2e/test_full_pipeline.py -v -k nemotron-mini-4b \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python --rebuild-engines

# Full diff validation
python3 tools/diff_logits.py --model nvidia/Nemotron-Mini-4B-Instruct --atol 1e-3 --battery --max-cache-length 256
python3 tools/diff_layers.py --model nvidia/Nemotron-Mini-4B-Instruct --atol 0.05
```
