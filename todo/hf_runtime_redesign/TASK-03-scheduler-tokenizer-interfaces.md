# TASK-03: IScheduler + ITokenizer interfaces

## Status: ready (after TASK-01)
## Phase: 1 (Foundation)
## Risk: low — ITokenizer wraps existing implementations; IScheduler is new but simple

## Goal

Define the shared component interfaces that pipelines compose — matching HF's
pattern where schedulers and tokenizers are interchangeable components.

## IScheduler interface

Currently: scheduler logic is inlined in each diffusion backend (3 copies of
flow-match Euler step). Extract into a proper interface.

```cpp
class IScheduler {
public:
    virtual ~IScheduler() = default;

    // Configure timestep schedule
    virtual void set_timesteps(int32_t num_steps) = 0;

    // Access timesteps (descending from ~1000 to ~0)
    virtual const std::vector<float>& timesteps() const = 0;

    // Single Euler step: latents += dt * velocity
    virtual void step(DeviceTensor& latents, const DeviceTensor& velocity,
                      int32_t step_index, cudaStream_t stream) = 0;
};

// Concrete: FlowMatchEulerScheduler (used by FLUX, Wan, Z-Image, SD3)
class FlowMatchEulerScheduler final : public IScheduler {
public:
    explicit FlowMatchEulerScheduler(float shift = 1.0f);
    void set_timesteps(int32_t num_steps) override;
    const std::vector<float>& timesteps() const override;
    void step(DeviceTensor& latents, const DeviceTensor& velocity,
              int32_t step_index, cudaStream_t stream) override;
private:
    float shift_;
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};
```

Note: `step()` operates on DeviceTensor directly (GPU-side axpy: latents += dt * velocity).
This avoids the current pattern of D2H → CPU scheduler step → H2D per denoiser step.

## ITokenizer interface

Currently: two implementations exist (`VocabTokenizer`, `HfPythonTokenizer`).
They have no shared interface — backends pick one directly.

```cpp
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int32_t> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int32_t>& ids) = 0;
};
```

Existing `VocabTokenizer` and `HfPythonTokenizer` already implement this
pattern — just need to formalize the interface and have them inherit from it.

## Factory helpers

```cpp
// Create scheduler from config string
std::unique_ptr<IScheduler> create_scheduler(const std::string& name, float shift = 1.0f);

// Create tokenizer from bundle (auto-detects vocab.txt vs tokenizer.json)
std::unique_ptr<ITokenizer> create_tokenizer_from_bundle(
    const BundleFile& bundle, const std::string& hf_python_path);
```

## Files to create/modify

- `include/trtf/runtime/scheduler.h` — IScheduler + FlowMatchEulerScheduler
- `src/runtime/trt/core/flow_match_euler_scheduler.cpp`
- `include/trtf/runtime/tokenizer.h` — ITokenizer interface
- Modify `src/tokenizer/vocab_tokenizer.h` — inherit from ITokenizer
- Modify `src/tokenizer/hf_python_tokenizer.h` — inherit from ITokenizer
- `tests/cpp/test_scheduler.cpp`

## Tests

- FlowMatchEuler: set_timesteps(28), verify timesteps match HF reference
- FlowMatchEuler: step() on GPU DeviceTensor, verify latent update
- ITokenizer: both VocabTokenizer and HfPythonTokenizer through interface

## Dependencies

TASK-01 (DeviceTensor for GPU-side scheduler step)
