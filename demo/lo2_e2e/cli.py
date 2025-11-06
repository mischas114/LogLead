#!/usr/bin/env python3
"""Command-line interface for declarative pipeline execution.

This CLI provides a simple interface to run pipelines defined in YAML
configuration files without any IF-based control flow.
"""

import argparse
import sys
from pathlib import Path

# Ensure core modules are imported to register components
from .core import registry
from .core.runner import run_pipeline_from_file
from . import steps  # Import to trigger registration
from .core import adapters  # Import to register adapters
from .core import explainers  # Import to register explainers


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run declarative ML pipelines without IF branching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a pipeline from config
  python -m demo.lo2_e2e.cli run --pipeline config/pipeline.yaml
  
  # Run with custom output
  python -m demo.lo2_e2e.cli run --pipeline config/pipeline.yaml --output results/
  
  # List available components
  python -m demo.lo2_e2e.cli list
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a pipeline from configuration")
    run_parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Path to pipeline configuration YAML file"
    )
    run_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (optional)"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List registered components")
    list_parser.add_argument(
        "--type",
        choices=["steps", "models", "explainers", "all"],
        default="all",
        help="Type of components to list"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "run":
        return run_command(args)
    elif args.command == "list":
        return list_command(args)
    
    return 0


def run_command(args):
    """Execute the run command."""
    pipeline_path = args.pipeline
    
    # Check if pipeline file exists
    if not Path(pipeline_path).exists():
        print(f"Error: Pipeline file not found: {pipeline_path}", file=sys.stderr)
        return 1
    
    print(f"Running pipeline from: {pipeline_path}")
    
    try:
        # Prepare initial context
        initial_context = {}
        if args.output:
            initial_context["output_dir"] = args.output
            Path(args.output).mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        context = run_pipeline_from_file(pipeline_path, initial_context)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        
        # Print summary
        if "explanations" in context:
            explanations = context["explanations"]
            n_local = len(explanations.get("local", []))
            print(f"\nGenerated {n_local} local explanations")
            print(f"Model: {explanations.get('model')}")
        
        if "predictions" in context:
            n_pred = len(context["predictions"])
            print(f"Generated {n_pred} predictions")
        
        return 0
        
    except Exception as e:
        print(f"\nError running pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def list_command(args):
    """Execute the list command."""
    show_type = args.type
    
    print("Registered Components")
    print("=" * 60)
    
    if show_type in ["steps", "all"]:
        print("\nSteps:")
        steps = registry.STEP_REGISTRY
        if steps:
            for name in sorted(steps.keys()):
                print(f"  - {name}")
        else:
            print("  (none registered)")
    
    if show_type in ["models", "all"]:
        print("\nModel Adapters:")
        models = registry.MODEL_REGISTRY
        if models:
            for name in sorted(models.keys()):
                cls = models[name]
                print(f"  - {name}: {cls.__name__}")
        else:
            print("  (none registered)")
    
    if show_type in ["explainers", "all"]:
        print("\nExplainers:")
        explainers = registry.EXPLAINER_REGISTRY
        if explainers:
            for name in sorted(explainers.keys()):
                cls = explainers[name]
                print(f"  - {name}: {cls.__name__}")
        else:
            print("  (none registered)")
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
