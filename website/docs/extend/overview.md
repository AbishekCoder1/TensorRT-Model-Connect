---
title: Extend the Project
---

import useBaseUrl from '@docusaurus/useBaseUrl';


Choose the smallest extension point that matches the change.

<figure className="trtmc-diagram trtmc-diagram--wide">
  <div className="trtmc-diagram__media">
    <img src={useBaseUrl('/img/diagrams/trtmc-extension-decision.svg')} alt="Extension decision tree" />
  </div>
  <figcaption>Start from the user-visible behavior, choose the owning abstraction, then add the narrowest validation that proves it.</figcaption>
</figure>

| Goal | Extension point |
| --- | --- |
| Build a new model that fits an existing runtime strategy | Add a Python family plugin. |
| Run a new task contract or state model | Add a C++ runtime strategy and plugin. |
| Add a new user-facing knob | Add a config schema and consume it in the owning unit. |
| Add a new CLI task | Add a command only when the public task cannot fit an existing command. |
| Add a new verifier | Add or extend an E2E harness plugin, comparator, or reference backend. |

Before adding a shared abstraction, verify that at least two real owners need it. The codebase favors local ownership and narrow files over central registries that every feature must edit.
