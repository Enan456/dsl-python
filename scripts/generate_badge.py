#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate SVG badge for Python-R comparison status.

Usage:
    python scripts/generate_badge.py --metrics metrics_summary.json --output badge.svg
"""

import argparse
import json


def generate_badge_svg(status: str, pass_rate: float) -> str:
    """Generate SVG badge based on status and pass rate.

    Args:
        status: Status string (passing, degraded, failing, no_data)
        pass_rate: Pass rate percentage

    Returns:
        SVG badge as string
    """
    # Color mapping
    colors = {
        'passing': '#4c1',
        'degraded': '#fe7d37',
        'failing': '#e05d44',
        'no_data': '#9f9f9f',
    }

    color = colors.get(status, '#9f9f9f')
    label = "Python-R Comparison"
    message = f"{pass_rate:.1f}% passing"

    # Calculate widths
    label_width = len(label) * 6 + 10
    message_width = len(message) * 6 + 10
    total_width = label_width + message_width

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h{label_width}v20H0z"/>
    <path fill="{color}" d="M{label_width} 0h{message_width}v20H{label_width}z"/>
    <path fill="url(#b)" d="M0 0h{total_width}v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width/2}" y="14">{label}</text>
    <text x="{label_width + message_width/2}" y="15" fill="#010101" fill-opacity=".3">{message}</text>
    <text x="{label_width + message_width/2}" y="14">{message}</text>
  </g>
</svg>'''

    return svg


def main():
    parser = argparse.ArgumentParser(
        description='Generate SVG badge for comparison status',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--metrics',
        type=str,
        required=True,
        help='Path to metrics summary JSON file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for SVG badge'
    )

    args = parser.parse_args()

    # Load metrics
    try:
        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        # Generate error badge
        svg = generate_badge_svg('no_data', 0.0)
        with open(args.output, 'w') as f:
            f.write(svg)
        return

    status = metrics.get('status', 'no_data')
    pass_rate = metrics.get('overall_pass_rate', 0.0)

    # Generate badge
    svg = generate_badge_svg(status, pass_rate)

    # Save badge
    with open(args.output, 'w') as f:
        f.write(svg)

    print(f"Badge generated: {args.output}")
    print(f"  Status: {status}")
    print(f"  Pass Rate: {pass_rate:.1f}%")


if __name__ == '__main__':
    main()
