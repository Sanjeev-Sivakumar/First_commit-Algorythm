#!/usr/bin/env python3
"""
KineticGuard - Train & Export Qwen2-VL LoRA Adapter (Phase 4 CLI)
Fine-tunes or exports the PEFT LoRA adapter for Qwen2-VL ergonomic analysis.

Features:
- Base model: Qwen/Qwen2-VL-2B-Instruct (frozen)
- LoRA attention projections: q_proj, v_proj
- Trainable parameters: < 0.1%
- Exports compliant Hugging Face PEFT adapter under models/qwen2_vl_ergonomic_lora/
- Supports fallback/mock mode for zero-cost offline execution

Usage:
  python train_lora.py
  python train_lora.py --epochs 3 --batch-size 2
  python train_lora.py --fallback
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.lora_trainer import Qwen2VLLoRATrainer


def main():
    parser = argparse.ArgumentParser(
        description="KineticGuard - Train / Export Qwen2-VL LoRA Adapter"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/qwen2_vl_ergonomic_lora",
        help="Directory to save LoRA adapter config and weights",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/ergonomic_dataset/annotations.json",
        help="Path to training annotations JSON",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per training step",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank dimension r (default: 8)",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor (default: 16)",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force lightweight offline PEFT export mode (no heavy GPU memory allocation)",
    )
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel(
            "[bold cyan]KineticGuard: Qwen2-VL LoRA Adapter Pipeline[/bold cyan]\n"
            "[dim]Phase 4: Parameter-Efficient Fine-Tuning (PEFT) for Ergonomics[/dim]",
            border_style="cyan",
        )
    )

    trainer = Qwen2VLLoRATrainer(
        output_adapter_dir=args.output_dir,
        dataset_path=args.dataset,
        r=args.rank,
        lora_alpha=args.alpha,
    )

    with console.status("[bold green]Configuring and exporting LoRA adapter weights..."):
        results = trainer.train_and_export(
            epochs=args.epochs,
            batch_size=args.batch_size,
            force_fallback=args.fallback,
        )

    table = Table(title="PEFT LoRA Training & Adapter Summary", border_style="green")
    table.add_column("Parameter / Config", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Base Model", "Qwen/Qwen2-VL-2B-Instruct [Frozen]")
    table.add_row("LoRA Target Modules", "q_proj, v_proj")
    table.add_row("LoRA Rank (r)", str(args.rank))
    table.add_row("LoRA Alpha (alpha)", str(args.alpha))
    table.add_row("Trainable LoRA Parameters", f"{results['total_lora_params']:,}")
    table.add_row("Trainable Parameter Ratio", results["trainable_percentage"])
    table.add_row("Execution Mode", results["mode"])
    table.add_row("Output Adapter Directory", results["adapter_dir"])

    console.print(table)

    console.print(
        Panel(
            f"[bold green]LoRA Adapter Successfully Prepared & Saved![/bold green]\n\n"
            f"- Configuration: [yellow]{results['config_path']}[/yellow]\n"
            f"- Safetensors Weights: [yellow]{results['weights_path']}[/yellow]\n\n"
            f"[dim]The LoRA adapter is ready for multimodal inference via `python qwen_analyze.py`.[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
