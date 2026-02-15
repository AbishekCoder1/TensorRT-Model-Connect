#!/usr/bin/env python3
"""
Orchestrator for parallel model family implementation via Claude Code subagents.

Usage:
    # Generate prompts and create branches for all Tier 1 families:
    python3 scripts/launch_model_agents.py --tier 1

    # Generate prompt for a single family:
    python3 scripts/launch_model_agents.py --family yi

    # Dry-run (print prompt without creating branches):
    python3 scripts/launch_model_agents.py --family yi --dry-run

    # After agents complete, merge all branches:
    python3 scripts/launch_model_agents.py --merge --tier 1

    # Generate Claude Code Task tool invocations:
    python3 scripts/launch_model_agents.py --tier 1 --task-tool
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelFamily:
    family: str           # lowercase, C++ namespace
    family_pascal: str    # PascalCase, class names
    model_type_prefix: str  # HF config.json model_type match prefix
    architectures: str    # HF architectures[0] string
    has_qk_norm: bool     # per-head Q/K RMSNorm
    hf_repo_id: str       # HuggingFace model repo for E2E validation
    tier: int             # 1 = standard dense decoder, 2 = custom graph builder
    notes: str = ""       # Tier 2: description of architectural difference


# ---------------------------------------------------------------------------
# Model family parameter table
# ---------------------------------------------------------------------------

TIER_1_FAMILIES = [
    ModelFamily(
        family="yi",
        family_pascal="Yi",
        model_type_prefix="yi",
        architectures="YiForCausalLM",
        has_qk_norm=False,
        hf_repo_id="01-ai/Yi-6B",
        tier=1,
    ),
    ModelFamily(
        family="mistral",
        family_pascal="Mistral",
        model_type_prefix="mistral",
        architectures="MistralForCausalLM",
        has_qk_norm=False,
        hf_repo_id="mistralai/Mistral-7B-v0.1",
        tier=1,
    ),
    ModelFamily(
        family="gemma",
        family_pascal="Gemma",
        model_type_prefix="gemma",
        architectures="GemmaForCausalLM",
        has_qk_norm=False,
        hf_repo_id="google/gemma-2b",
        tier=1,
    ),
    ModelFamily(
        family="internlm",
        family_pascal="InternLM",
        model_type_prefix="internlm",
        architectures="InternLMForCausalLM",
        has_qk_norm=False,
        hf_repo_id="internlm/internlm2-7b",
        tier=1,
    ),
    ModelFamily(
        family="deepseek",
        family_pascal="DeepSeek",
        model_type_prefix="deepseek",
        architectures="DeepseekForCausalLM",
        has_qk_norm=False,
        hf_repo_id="deepseek-ai/deepseek-llm-7b-base",
        tier=1,
    ),
    ModelFamily(
        family="baichuan",
        family_pascal="Baichuan",
        model_type_prefix="baichuan",
        architectures="BaichuanForCausalLM",
        has_qk_norm=False,
        hf_repo_id="baichuan-inc/Baichuan2-7B-Base",
        tier=1,
    ),
]

TIER_2_FAMILIES = [
    ModelFamily(
        family="phi",
        family_pascal="Phi",
        model_type_prefix="phi",
        architectures="PhiForCausalLM",
        has_qk_norm=False,
        hf_repo_id="microsoft/phi-2",
        tier=2,
        notes="Partial rotary embedding, parallel attention+MLP",
    ),
    ModelFamily(
        family="falcon",
        family_pascal="Falcon",
        model_type_prefix="falcon",
        architectures="FalconForCausalLM",
        has_qk_norm=False,
        hf_repo_id="tiiuae/falcon-7b",
        tier=2,
        notes="Multi-query attention, LayerNorm (not RMSNorm)",
    ),
    ModelFamily(
        family="gptneox",
        family_pascal="GPTNeoX",
        model_type_prefix="gpt_neox",
        architectures="GPTNeoXForCausalLM",
        has_qk_norm=False,
        hf_repo_id="EleutherAI/pythia-1.4b",
        tier=2,
        notes="Parallel attention, rotary on partial dims",
    ),
    ModelFamily(
        family="stablelm",
        family_pascal="StableLM",
        model_type_prefix="stablelm",
        architectures="StableLmForCausalLM",
        has_qk_norm=False,
        hf_repo_id="stabilityai/stablelm-3b-4e1t",
        tier=2,
        notes="Different RoPE variant (NTK-aware)",
    ),
]

ALL_FAMILIES = TIER_1_FAMILIES + TIER_2_FAMILIES


def get_families(tier: Optional[int] = None, family_name: Optional[str] = None) -> List[ModelFamily]:
    """Filter families by tier or name."""
    if family_name:
        matches = [f for f in ALL_FAMILIES if f.family == family_name]
        if not matches:
            available = ", ".join(f.family for f in ALL_FAMILIES)
            print(f"Unknown family '{family_name}'. Available: {available}", file=sys.stderr)
            sys.exit(1)
        return matches
    if tier is not None:
        return [f for f in ALL_FAMILIES if f.tier == tier]
    return ALL_FAMILIES


def hf_local_dir(repo_id: str) -> str:
    """Convert HF repo_id to local directory name (org__model)."""
    return repo_id.replace("/", "__")


def generate_qk_norm_assertions(family: str, has_qk_norm: bool) -> str:
    """Generate the QK norm test assertions block (C++ code)."""
    if has_qk_norm:
        return (
            '            if (layer0.q_norm.empty())\n'
            '            {\n'
            f'                std::cerr << "expected {family} q_norm to be populated (has QK norm)" << std::endl;\n'
            '                return 1;\n'
            '            }\n'
            '            if (layer0.k_norm.empty())\n'
            '            {\n'
            f'                std::cerr << "expected {family} k_norm to be populated (has QK norm)" << std::endl;\n'
            '                return 1;\n'
            '            }'
        )
    else:
        return (
            '            if (!layer0.q_norm.empty())\n'
            '            {\n'
            f'                std::cerr << "expected {family} q_norm to be empty (no QK norm), got size " << layer0.q_norm.size() << std::endl;\n'
            '                return 1;\n'
            '            }\n'
            '            if (!layer0.k_norm.empty())\n'
            '            {\n'
            f'                std::cerr << "expected {family} k_norm to be empty (no QK norm), got size " << layer0.k_norm.size() << std::endl;\n'
            '                return 1;\n'
            '            }'
        )


def generate_qk_norm_test_comment(has_qk_norm: bool) -> str:
    if has_qk_norm:
        return "WITH q_norm/k_norm (per-head QK RMSNorm)"
    return "WITHOUT q_norm/k_norm"


def generate_prompt(fam: ModelFamily) -> str:
    """Generate the concrete agent prompt from the template using simple string replacement."""
    template_path = Path(__file__).parent / "agents" / "implement-model-family.md"
    template = template_path.read_text()

    local_dir = hf_local_dir(fam.hf_repo_id)
    prefix_len = str(len(fam.model_type_prefix))
    qk_assertions = generate_qk_norm_assertions(fam.family, fam.has_qk_norm)
    qk_test_comment = generate_qk_norm_test_comment(fam.has_qk_norm)

    # Simple string replacements — order matters (longer patterns first to avoid partial matches)
    replacements = [
        ("__model_type_prefix_len__", prefix_len),
        ("__model_type_prefix__", fam.model_type_prefix),
        ("__qk_norm_test_comment__", qk_test_comment),
        ("__qk_norm_assertions__", qk_assertions),
        ("__has_qk_norm__", "true" if fam.has_qk_norm else "false"),
        ("__hf_local_dir__", local_dir),
        ("__hf_repo_id__", fam.hf_repo_id),
        ("__architectures__", fam.architectures),
        ("__Family__", fam.family_pascal),
        ("__family__", fam.family),
    ]

    prompt = template
    for old, new in replacements:
        prompt = prompt.replace(old, new)

    # Add tier 2 notes if applicable
    if fam.tier == 2:
        tier2_note = (
            f"\n\n## IMPORTANT: Tier 2 -- Custom Graph Builder Required\n\n"
            f"This model family has architectural differences from the standard decoder:\n"
            f"**{fam.notes}**\n\n"
            f"You MUST create a custom `ITrtGraphBuilder` instead of using "
            f"`StandardDecoderGraphBuilder`. See the 'Tier 2 Extension' section at the end "
            f"of this prompt for the template and available TRT graph ops.\n\n"
            f"Read `src/model/standard_decoder_graph_builder.cpp` to understand the "
            f"standard pattern, then modify it for this family's architecture.\n"
        )
        insert_point = prompt.find("## Step 0:")
        if insert_point != -1:
            prompt = prompt[:insert_point] + tier2_note + "\n" + prompt[insert_point:]

    return prompt


def create_branch(family: str, dry_run: bool = False) -> bool:
    """Create a git branch for the family (without switching to it)."""
    branch = f"model/{family}"
    if dry_run:
        print(f"  [dry-run] Would create branch: {branch}")
        return True

    # Check if branch already exists
    result = subprocess.run(
        ["git", "rev-parse", "--verify", branch],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Branch {branch} already exists.")
        return True

    # Create from master
    subprocess.run(["git", "branch", branch, "master"], check=True)
    print(f"  Created branch: {branch}")
    return True


def write_concrete_prompt(fam: ModelFamily, output_dir: Path, dry_run: bool = False) -> Path:
    """Write the concrete (substituted) prompt to a file."""
    output_path = output_dir / f"{fam.family}.md"
    prompt = generate_prompt(fam)

    if dry_run:
        print(f"  [dry-run] Would write prompt to: {output_path}")
        print(f"  Prompt length: {len(prompt)} chars, {prompt.count(chr(10))} lines")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt)
    print(f"  Wrote prompt: {output_path} ({len(prompt)} chars)")
    return output_path


def merge_branches(families: List[ModelFamily], dry_run: bool = False):
    """Merge all family branches into master."""
    if dry_run:
        print("\n[dry-run] Would merge these branches into master:")
        for fam in families:
            print(f"  model/{fam.family}")
        return

    subprocess.run(["git", "checkout", "master"], check=True)

    for fam in families:
        branch = f"model/{fam.family}"
        print(f"\nMerging {branch} into master...")
        result = subprocess.run(
            ["git", "merge", branch, "--no-edit",
             "-m", f"Merge {fam.family_pascal} model family from {branch}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  CONFLICT merging {branch}. Resolve manually:")
            print(f"    git merge {branch}")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
        print(f"  Merged {branch} successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate parallel model family implementation agents")
    parser.add_argument("--tier", type=int, choices=[1, 2],
                        help="Select families by tier (1=standard, 2=custom)")
    parser.add_argument("--family", type=str,
                        help="Select a single family by name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without making changes")
    parser.add_argument("--merge", action="store_true",
                        help="Merge completed family branches into master")
    parser.add_argument("--task-tool", action="store_true",
                        help="Print Claude Code Task tool invocations")
    parser.add_argument("--prompt-only", action="store_true",
                        help="Only generate and print the prompt (no branches)")
    parser.add_argument("--output-dir", type=str,
                        default="scripts/agents/prompts",
                        help="Directory for generated prompt files")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    families = get_families(tier=args.tier, family_name=args.family)

    if not families:
        print("No families selected.", file=sys.stderr)
        sys.exit(1)

    if args.merge:
        merge_branches(families, dry_run=args.dry_run)
        return

    if args.prompt_only and len(families) == 1:
        print(generate_prompt(families[0]))
        return

    print(f"Selected {len(families)} model families:")
    for fam in families:
        tier_label = "Tier 1 (standard)" if fam.tier == 1 else f"Tier 2 ({fam.notes})"
        print(f"  {fam.family_pascal:12s} model_type={fam.model_type_prefix:12s} {tier_label}")
    print()

    # Step 1: Create branches
    if not args.prompt_only:
        print("Creating branches...")
        for fam in families:
            create_branch(fam.family, dry_run=args.dry_run)
        print()

    # Step 2: Generate concrete prompts
    print("Generating agent prompts...")
    output_dir = Path(args.output_dir)
    prompt_paths = {}
    for fam in families:
        path = write_concrete_prompt(fam, output_dir, dry_run=args.dry_run)
        prompt_paths[fam.family] = path
    print()

    # Step 3: Print task tool invocations if requested
    if args.task_tool:
        print("=" * 72)
        print("Claude Code Task tool invocations")
        print("Copy these into your Claude Code conversation to launch agents:")
        print("=" * 72)
        for fam in families:
            prompt_file = prompt_paths.get(fam.family, output_dir / f"{fam.family}.md")
            print(f'\n--- {fam.family_pascal} ---')
            print(f'Use Task tool with:')
            print(f'  description: "Implement {fam.family_pascal} model family"')
            print(f'  subagent_type: "general-purpose"')
            print(f'  prompt: Read the file at {prompt_file} and follow ALL instructions.')
            print(f'  run_in_background: true')
        print()

    # Print next steps
    print("=" * 72)
    print("Next steps:")
    print("=" * 72)
    if args.dry_run:
        print("  Re-run without --dry-run to execute.")
    else:
        print("  1. For each family, launch a Claude Code agent with the generated prompt.")
        print("     Prompt files are at: scripts/agents/prompts/<family>.md")
        print()
        print("  2. Each agent works on its own branch (model/<family>).")
        print("     Agents can run in parallel without conflicts.")
        print()
        print("  3. After all agents complete, merge branches:")
        tier_flag = f"--tier {args.tier}" if args.tier else "--family <name>"
        print(f"     python3 scripts/launch_model_agents.py --merge {tier_flag}")
        print()
        print("  4. Run full test suite after merge:")
        print("     docker exec trtf-dev bash -c 'cmake --build build-container-phase1 -j'")
        print("     docker exec trtf-dev bash -c 'ctest --test-dir build-container-phase1 --output-on-failure'")


if __name__ == "__main__":
    main()
