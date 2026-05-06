---
title: Learning Path
description: A course-style path for learning TensorRT-Model-Connect from inference fundamentals to extension work.
---

import useBaseUrl from '@docusaurus/useBaseUrl';


Use this page like a course handout. Each stage tells you what to read, what to do, what evidence to record, and what you should be able to explain before moving on.

<div className="trtmc-handout-meta">
  <div>
    <strong>Audience</strong>
    <span>Readers new to inference and TensorRT deployment.</span>
  </div>
  <div>
    <strong>Method</strong>
    <span>Concept, command, inspection, explanation, validation.</span>
  </div>
  <div>
    <strong>Outcome</strong>
    <span>Understand the project well enough to use and extend it.</span>
  </div>
  <div>
    <strong>Artifact</strong>
    <span>A built `.trtfb` bundle and a written learning log.</span>
  </div>
</div>

<figure className="trtmc-diagram trtmc-diagram--wide">
  <div className="trtmc-diagram__media">
    <img src={useBaseUrl('/img/diagrams/trtmc-course-map.svg')} alt="Course-style learning map for TensorRT-Model-Connect" />
  </div>
  <figcaption>Each stage has a reading target, required task, and proof point so readers can check understanding as they go.</figcaption>
</figure>

## Information Boxes

The handouts use consistent information categories:

:::info Required reading
Read this before attempting the task. The goal is to build the right mental model before running commands.
:::

:::danger Required task
Complete this task and keep the command output or observation in your learning log.
:::

:::tip Progress check
Use this to decide whether you are ready for the next stage.
:::

:::note Further reading
Use these links when you need a deeper explanation, but do not block the main path on them.
:::

:::warning Common trap
This marks a misunderstanding that usually sends debugging in the wrong direction.
:::

## Learning Log

Create a learning log as you work through the course. The point is not paperwork; it is to make sure you can reconstruct the reasoning instead of only replaying commands.

<div className="trtmc-log-template">
  <p><strong>Stage:</strong> Name of the stage.</p>
  <p><strong>Command or file:</strong> The command you ran or source file you read.</p>
  <p><strong>Observation:</strong> What changed, what output mattered, or what file proved the point.</p>
  <p><strong>Explanation:</strong> One or two sentences in your own words.</p>
  <p><strong>Next question:</strong> The next thing you still cannot explain.</p>
</div>

## Course Outcomes

After completing the path, you should be able to:

- Explain what inference is and how it differs from training.
- Describe why TensorRT engines are build artifacts, not raw checkpoints.
- Build and inspect a `.trtfb` bundle.
- Trace a request from `trtmc::load()` to a concrete `IPipeline`.
- Explain the difference between Python family plugins and C++ runtime strategies.
- Decide where to add a model family, runtime plugin, pipeline, config schema, or test.

## Stage F1: Learn the Vocabulary

:::info Required reading

- [Glossary](getting-started/glossary.md)
- [Inference Fundamentals](getting-started/inference-fundamentals.md)

:::

:::danger Required task
Write a one-paragraph explanation of model checkpoints, tensors, tokens, logits, TensorRT engines, prefill, decode, KV cache, and bundles. Do not use source code terms yet.
:::

:::tip Progress check
You are ready to move on when you can draw the path from prompt text to `TextResult` and explain why a `.trtfb` bundle is not the same thing as a HuggingFace checkpoint.
:::

<details>
<summary>Further reading</summary>

- [Architecture Overview](architecture/overview.md)
- [Bundle Format](architecture/bundle-format.md)

</details>

## Stage F2: Build and Run One Model

:::info Required reading

- [Environment and First Repro](getting-started/environment-and-repro.md)
- [Quick Start](getting-started/quick-start.md)
- [Beginner Tutorial - Inspect Bundles](tutorials/beginner/inspect-bundles.md)
- [Beginner Tutorial - Text Generation](tutorials/beginner/text-generation.md)

:::

:::danger Required task
Build one text-generation bundle, inspect it, and run deterministic generation. Record the exact `family`, `runtime_strategy`, engine section names, tokenizer assets, precision, and TensorRT metadata.
:::

:::tip Progress check
You are ready to move on when you can explain which part of the output came from the Python builder, which part came from the bundle, and which part came from the C++ runtime.
:::

:::warning Common trap
Do not treat a successful text response as the whole validation. You also need to inspect the bundle and confirm the runtime strategy that produced the response.
:::

## Stage E1: Understand the System

:::info Required reading

- [Architecture Overview](architecture/overview.md)
- [Runtime Plugins](architecture/runtime-plugins.md)
- [Build System](architecture/build-system.md)

:::

:::danger Required task
Trace `trtmc::load()` through `PipelineFactory`, `PipelineRegistry`, `IPipelinePlugin`, `IBackend`, and the concrete `IPipeline`. Add the source paths you inspected to your learning log.
:::

:::tip Progress check
You are ready to move on when you can say why the runtime dispatches through `runtime_strategy` instead of a central switch on model names.
:::

## Stage E2: Learn the Source Units

:::info Required reading

- [Unit Design Overview](unit-design/overview.md)
- [Building Blocks](unit-design/building-blocks.md)
- [Python Builder Units](unit-design/python-builder.md)
- [C++ Runtime Units](unit-design/cpp-runtime.md)

:::

:::danger Required task
Pick a hypothetical new decoder model and identify the first three files or modules you would inspect before editing. Then pick a hypothetical new request-time task and identify the runtime units you would inspect.
:::

:::tip Progress check
You are ready to move on when you can distinguish model knowledge, artifact evidence, runtime behavior, and user-facing API changes.
:::

## Stage D1: Compare Modalities

:::info Required reading

- [Intermediate Tutorial - Multimodal and Speech](tutorials/intermediate/multimodal-and-speech.md)
- [Intermediate Tutorial - Diffusion and Time-Series](tutorials/intermediate/diffusion-and-time-series.md)
- [Features - Model Families](features/model-families.md)
- [Features - Runtime Strategies](features/runtime-strategies.md)

:::

:::danger Required task
For text, audio, image, diffusion, time-series, segmentation, and detection, name the preprocessing, engine components, postprocessing, and public `IPipeline` method.
:::

:::tip Progress check
You are ready to move on when you can explain which differences are task differences and which differences are only model-family conversion details.
:::

## Stage C1: Extend and Validate

:::info Required reading

- [Extend Overview](extend/overview.md)
- [Add a Model Family](extend/add-model-family.md)
- [Add a Runtime Strategy](extend/add-runtime-strategy.md)
- [Advanced Tutorial - Validation and Benchmarking](tutorials/advanced/validation-and-benchmarking.md)

:::

:::danger Required task
Fill out the extension decision tree for your change. Name the builder tests, C++ tests, E2E manifest, and documentation page that should prove the behavior.
:::

:::tip Progress check
You finish the course when you can explain the implementation plan before opening an editor, and the plan names the owning abstraction for every change.
:::

<figure className="trtmc-diagram trtmc-diagram--wide">
  <div className="trtmc-diagram__media">
    <img src={useBaseUrl('/img/diagrams/trtmc-extension-decision.svg')} alt="Extension decision tree" />
  </div>
  <figcaption>Pick the smallest ownership boundary that can support the behavior, then prove it with the matching tests.</figcaption>
</figure>
