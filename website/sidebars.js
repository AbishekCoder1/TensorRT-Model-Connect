const sidebars = {
  docs: [
    'intro',
    'learning-path',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/glossary',
        'getting-started/environment-and-repro',
        'getting-started/inference-fundamentals',
        'getting-started/quick-start',
        'getting-started/installation',
        'getting-started/build-and-run',
        'getting-started/model-support'
      ]
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/beginner/inspect-bundles',
        'tutorials/beginner/text-generation',
        'tutorials/intermediate/multimodal-and-speech',
        'tutorials/intermediate/diffusion-and-time-series',
        'tutorials/advanced/quantization-and-runtime-knobs',
        'tutorials/advanced/validation-and-benchmarking'
      ]
    },
    {
      type: 'category',
      label: 'API Manual',
      items: [
        'api/overview',
        'api/python-builder',
        'api/cli-reference',
        'api/cpp-api'
      ]
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
        'architecture/bundle-format',
        'architecture/runtime-plugins',
        'architecture/build-system'
      ]
    },
    {
      type: 'category',
      label: 'Unit Design',
      items: [
        'unit-design/overview',
        'unit-design/building-blocks',
        'unit-design/python-builder',
        'unit-design/cpp-runtime',
        'unit-design/testing'
      ]
    },
    {
      type: 'category',
      label: 'Features',
      items: [
        'features/model-families',
        'features/runtime-strategies',
        'features/quantization',
        'features/config-and-backends'
      ]
    },
    {
      type: 'category',
      label: 'Extend',
      items: [
        'extend/overview',
        'extend/add-model-family',
        'extend/add-runtime-strategy',
        'extend/add-config-schema'
      ]
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/source-layout',
        'reference/testing',
        'reference/documentation-research'
      ]
    }
  ]
};

module.exports = sidebars;
