"""
Visualize tokenizer output as a colored HTML page.

Usage:
    # Interactive: type text and see tokenization
    python -m scripts.tok_viz

    # Tokenize a string
    python -m scripts.tok_viz "fn main() { println!(\"Hello, world!\"); }"

    # Tokenize a file
    python -m scripts.tok_viz --file example.rs

    # Save HTML output
    python -m scripts.tok_viz --file example.rs -o tokenizer_viz.html
"""

import os
import argparse
import html
import colorsys

from etude.tokenizer import get_tokenizer

# Generate a palette of visually distinct, pastel background colors
def _generate_palette(n):
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio for spread
        r, g, b = colorsys.hls_to_rgb(hue, 0.85, 0.65)
        colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
    return colors

PALETTE = _generate_palette(64)


def tokenize_to_html(text, tokenizer):
    """Tokenize text and return an HTML string with colored spans."""
    token_ids = tokenizer.encode(text)
    parts = []

    parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 14px;
       background: #1e1e1e; color: #d4d4d4; padding: 24px; line-height: 1.6; }
.token { padding: 2px 1px; border-radius: 3px; cursor: pointer; position: relative; }
.token:hover { outline: 2px solid #fff; z-index: 1; }
.token:hover::after {
    content: attr(data-info);
    position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
    background: #333; color: #fff; padding: 4px 8px; border-radius: 4px;
    font-size: 11px; white-space: nowrap; z-index: 10; pointer-events: none;
}
.stats { background: #2d2d2d; padding: 16px; border-radius: 8px; margin-bottom: 20px; }
.stats span { margin-right: 24px; }
.legend { color: #888; font-size: 12px; margin-bottom: 8px; }
pre { white-space: pre-wrap; word-wrap: break-word; }
</style></head><body>
""")

    # Stats
    text_bytes = len(text.encode('utf-8'))
    n_tokens = len(token_ids)
    ratio = text_bytes / n_tokens if n_tokens > 0 else 0
    parts.append(f'<div class="stats">')
    parts.append(f'<span><b>Tokens:</b> {n_tokens:,}</span>')
    parts.append(f'<span><b>Bytes:</b> {text_bytes:,}</span>')
    parts.append(f'<span><b>Compression:</b> {ratio:.2f} bytes/token</span>')
    parts.append(f'<span><b>Vocab size:</b> {tokenizer.get_vocab_size():,}</span>')
    parts.append(f'</div>')
    parts.append(f'<div class="legend">Hover over tokens to see ID and byte representation</div>')
    parts.append('<pre>')

    for i, token_id in enumerate(token_ids):
        token_str = tokenizer.decode([token_id])
        token_bytes = token_str.encode('utf-8')
        hex_repr = ' '.join(f'{b:02x}' for b in token_bytes)

        color = PALETTE[token_id % len(PALETTE)]
        escaped = html.escape(token_str).replace('\n', '↵\n').replace(' ', '·')
        info = f"id={token_id} | bytes=[{hex_repr}] | &#39;{html.escape(token_str)}&#39;"

        parts.append(
            f'<span class="token" style="background:{color};color:#000" '
            f'data-info="{info}">{escaped}</span>'
        )

    parts.append('</pre></body></html>')
    return ''.join(parts)


SAMPLE_TEXTS = {
    "rust": '''use std::collections::HashMap;

fn word_count(text: &str) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

fn main() {
    let text = "hello world hello rust world";
    let counts = word_count(text);
    for (word, count) in &counts {
        println!("{word}: {count}");
    }
}
''',
    "english": '''Photosynthesis is a photochemical energy transduction process in which
light-harvesting pigment-protein complexes within the thylakoid membranes
absorb photons and initiate charge separation at the reaction center,
driving the linear electron transport chain from water to NADP+.
''',
    "mixed": '''# Rust Ownership Model

In Rust, every value has a single "owner" variable. When the owner goes
out of scope, the value is dropped:

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    // println!("{}", s1); // ERROR: s1 no longer valid
    println!("{}", s2); // OK
}
```

This prevents double-free bugs at compile time — no garbage collector needed.
''',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize tokenizer output")
    parser.add_argument("text", nargs="?", default=None, help="Text to tokenize")
    parser.add_argument("--file", "-f", type=str, default=None, help="File to tokenize")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output HTML file (default: open in browser)")
    parser.add_argument("--sample", "-s", type=str, choices=list(SAMPLE_TEXTS.keys()),
                        default=None, help="Use a built-in sample text")
    args = parser.parse_args()

    tokenizer = get_tokenizer()

    # Get input text
    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    elif args.sample:
        text = SAMPLE_TEXTS[args.sample]
    else:
        # Default: show all samples
        text = "\n---\n\n".join(f"=== {name} ===\n\n{t}" for name, t in SAMPLE_TEXTS.items())

    html_output = tokenize_to_html(text, tokenizer)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(html_output)
        print(f"Saved to {args.output}")
    else:
        # Write to temp file and open in browser
        import tempfile
        import webbrowser
        with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
            f.write(html_output)
            tmp_path = f.name
        print(f"Opening {tmp_path}")
        webbrowser.open(f'file://{tmp_path}')
