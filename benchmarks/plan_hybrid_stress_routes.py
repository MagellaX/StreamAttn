"""Emit reproducible commands for late-layer hybrid stress route experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_seed_only_stress_attribution import QWEN3B_POLICY_BY_LAYER


DEFAULT_OUTPUT_DIR = "artifacts/gate0/qwen25_3b_32k_b8_hybrid_routes"


@dataclass(frozen=True)
class HybridRouteSpec:
    name: str
    seed_layers: tuple[int, ...]
    dynamic_layers: tuple[int, ...] = ()
    dynamic_profile: str = ""

    @property
    def policy_names(self) -> str:
        return ",".join(QWEN3B_POLICY_BY_LAYER[layer] for layer in self.seed_layers)


def build_hybrid_specs() -> List[HybridRouteSpec]:
    """Return the focused L27-exact stress route set.

    The base stress hypothesis is that L27 is not safely seed-only on stress
    rows, while L26 may still be admissible as seed or dynamic seed.
    """

    return [
        HybridRouteSpec(
            name="stress_l27_exact_l26_seed",
            seed_layers=(0, 14, 16, 24, 26, 35),
        ),
        HybridRouteSpec(
            name="stress_l27_exact_l26_dynamic_extreme4",
            seed_layers=(0, 14, 16, 24, 26, 35),
            dynamic_layers=(26,),
            dynamic_profile="support_extreme4_mean_refine32",
        ),
        HybridRouteSpec(
            name="stress_l27_exact_l26_dynamic_qk",
            seed_layers=(0, 14, 16, 24, 26, 35),
            dynamic_layers=(26,),
            dynamic_profile="qk_block_max",
        ),
        HybridRouteSpec(
            name="stress_l26_l27_exact",
            seed_layers=(0, 14, 16, 24, 35),
        ),
        HybridRouteSpec(
            name="stress_l24_l26_l27_exact",
            seed_layers=(0, 14, 16, 35),
        ),
    ]


def _base_modal_command(
    *,
    spec: HybridRouteSpec,
    output_dir: str,
    steps: int,
    use_hf_token_secret: bool,
) -> List[str]:
    output_path = str(Path(output_dir) / f"{spec.name}_{steps}step_h100.json")
    cmd = [
        "modal",
        "run",
        "benchmarks\\modal_seed_only_route_bundle_decode.py",
        "--policy-names",
        spec.policy_names,
        "--prompt-file",
        "benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl",
        "--prompt-truncation-side",
        "left",
        "--max-prompts",
        "8",
        "--batch-size",
        "8",
        "--max-seq",
        "32768",
        "--steps",
        str(int(steps)),
        "--warmup-steps",
        "2",
        "--native-routed-cache",
        "--fused-rope-append-seed",
        "--packed-qkv-projection",
        "--margin-forensics",
        "--output-json",
        output_path,
    ]
    if spec.dynamic_layers:
        cmd.extend(["--dynamic-selector-layers", ",".join(str(layer) for layer in spec.dynamic_layers)])
        cmd.extend(["--dynamic-selector-profile", spec.dynamic_profile])
    if use_hf_token_secret:
        cmd.append("--use-hf-token-secret")
    return cmd


def build_plan(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    steps: int = 32,
    use_hf_token_secret: bool = False,
) -> Dict[str, Any]:
    routes = []
    for spec in build_hybrid_specs():
        routes.append(
            {
                "name": spec.name,
                "seed_layers": list(spec.seed_layers),
                "exact_layers": [layer for layer in (0, 14, 16, 24, 26, 27, 35) if layer not in spec.seed_layers],
                "dynamic_layers": list(spec.dynamic_layers),
                "dynamic_profile": spec.dynamic_profile,
                "policy_names": spec.policy_names,
                "command": _base_modal_command(
                    spec=spec,
                    output_dir=output_dir,
                    steps=steps,
                    use_hf_token_secret=use_hf_token_secret,
                ),
            }
        )
    return {
        "schema": "streamattn.hybrid_stress_route_plan.v1",
        "hypothesis": "L27 exact by default; test whether L26 can remain seed/dynamic on stress rows.",
        "output_dir": output_dir,
        "steps": int(steps),
        "use_hf_token_secret": bool(use_hf_token_secret),
        "routes": routes,
    }


def print_plan(plan: Dict[str, Any]) -> None:
    print("Hybrid stress route plan")
    print(f"  steps: {plan['steps']}")
    print(f"  output_dir: {plan['output_dir']}")
    print(f"  use_hf_token_secret: {plan['use_hf_token_secret']}")
    for route in plan["routes"]:
        print()
        print(route["name"])
        print(f"  seed:    {route['seed_layers']}")
        print(f"  exact:   {route['exact_layers']}")
        if route["dynamic_layers"]:
            print(f"  dynamic: {route['dynamic_layers']} profile={route['dynamic_profile']}")
        print("  command:")
        print("    " + " ".join(route["command"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--use-hf-token-secret", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    plan = build_plan(
        output_dir=args.output_dir,
        steps=args.steps,
        use_hf_token_secret=args.use_hf_token_secret,
    )
    print_plan(plan)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
