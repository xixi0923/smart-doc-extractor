"""
Smart Doc Extractor — CLI Entry Point
======================================
Command-line interface for document information extraction.

Usage:
    python main.py extract <image_path> [--type TYPE] [--output DIR]
    python main.py batch <directory> [--type TYPE] [--output DIR]
    python main.py serve [--host HOST] [--port PORT]
    python main.py config [save|show] [--path FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from config import AppConfig
from src import __version__
from src.utils.logger import get_logger, setup_logging


def cmd_extract(args, config: AppConfig) -> None:
    """Process a single document image."""
    from src.pipeline.extractor import DocumentExtractor
    from src.utils.image_utils import load_image
    from src.visualization.result_viewer import ResultViewer

    setup_logging(level=config.log.level, log_format=config.log.format)
    logger = get_logger("cli")

    # Override document type if specified
    if args.type:
        config.extraction.document_type = args.type

    output_dir = Path(args.output) if args.output else Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = DocumentExtractor(config)
    viewer = ResultViewer()

    logger.info(f"Extracting: {args.image}")
    result = extractor.process(args.image)

    if not result.success:
        logger.error(f"Extraction failed: {result.error}")
        sys.exit(1)

    # Save annotated image
    original = load_image(args.image)
    image_name = Path(args.image).stem
    annotated_path = output_dir / f"{image_name}_annotated.png"
    viewer.save_annotated(original, result, str(annotated_path))

    # Generate HTML report
    report_path = output_dir / f"{image_name}_report.html"
    viewer.generate_html_report(
        result, str(report_path), str(annotated_path)
    )

    # Save JSON result
    json_path = output_dir / f"{image_name}_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Smart Doc Extractor — Result")
    print(f"{'='*60}")
    print(f"Source:       {result.source_path}")
    print(f"Image size:   {result.image_size[1]}x{result.image_size[0]}")
    print(f"Doc type:     {result.extraction_result.document_type}")
    print(f"Fields:       {result.extraction_result.total_fields}")
    print(f"High conf:    {result.extraction_result.high_confidence_fields}")
    print(f"Total time:   {result.total_time_ms:.0f}ms")
    print(f"\nExtracted Fields:")

    for f in result.extraction_result.fields:
        conf_pct = f"{f.confidence * 100:.0f}%"
        print(f"  {f.display_name:12s} | {f.value[:40]:40s} | {conf_pct:>5s}")

    print(f"\nOutput files:")
    print(f"  JSON:     {json_path}")
    print(f"  Image:    {annotated_path}")
    print(f"  Report:   {report_path}")
    print(f"{'='*60}")


def cmd_batch(args, config: AppConfig) -> None:
    """Process a batch of document images."""
    from src.pipeline.extractor import DocumentExtractor

    setup_logging(level=config.log.level, log_format=config.log.format)
    logger = get_logger("cli")

    if args.type:
        config.extraction.document_type = args.type

    output_dir = Path(args.output) if args.output else Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images in directory
    input_dir = Path(args.directory)
    extensions = set(config.image.allowed_formats)
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_paths:
        logger.error(f"No images found in {args.directory}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images to process")

    extractor = DocumentExtractor(config)

    def progress_callback(current: int, total: int, result):
        status = "OK" if result.success else "FAIL"
        print(f"  [{current}/{total}] {status} - {Path(result.source_path).name}")

    print(f"\nProcessing {len(image_paths)} documents...")
    results = extractor.process_batch(
        [str(p) for p in image_paths],
        callback=progress_callback,
    )

    # Summary
    success = sum(1 for r in results if r.success)
    print(f"\nBatch complete: {success}/{len(image_paths)} successful")


def cmd_serve(args, config: AppConfig) -> None:
    """Start the REST API server."""
    from src.api.server import run_server

    if args.host:
        config.api.host = args.host
    if args.port:
        config.api.port = args.port

    run_server(config)


def cmd_config(args, config: AppConfig) -> None:
    """Show or save configuration."""
    if args.action == "show":
        print(config.to_json())
    elif args.action == "save":
        path = args.path or "config.json"
        config.save(path)
        print(f"Config saved to: {path}")
    else:
        print("Usage: config [show|save] [--path FILE]")


def main():
    parser = argparse.ArgumentParser(
        prog="smart-doc-extractor",
        description="Smart Doc Extractor — Intelligent document information extraction",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # extract command
    extract_parser = subparsers.add_parser("extract", help="Extract from a single image")
    extract_parser.add_argument("image", help="Path to input image")
    extract_parser.add_argument("--type", help="Document type (auto/invoice/receipt/contract)")
    extract_parser.add_argument("--output", help="Output directory")
    extract_parser.set_defaults(func=cmd_extract)

    # batch command
    batch_parser = subparsers.add_parser("batch", help="Process a directory of images")
    batch_parser.add_argument("directory", help="Directory containing images")
    batch_parser.add_argument("--type", help="Document type")
    batch_parser.add_argument("--output", help="Output directory")
    batch_parser.set_defaults(func=cmd_batch)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument("--host", help="Host address")
    serve_parser.add_argument("--port", type=int, help="Port number")
    serve_parser.set_defaults(func=cmd_serve)

    # config command
    config_parser = subparsers.add_parser("config", help="Show or save configuration")
    config_parser.add_argument("action", choices=["show", "save"], help="show or save")
    config_parser.add_argument("--path", help="File path for save")
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = AppConfig()
    args.func(args, config)


if __name__ == "__main__":
    main()
