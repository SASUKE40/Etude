"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import torch
from etude.common import compute_init, autodetect_device_type
from etude.engine import Engine
from etude.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: base|sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")
im_start = tokenizer.encode_special("<|im_start|>")
im_end = tokenizer.encode_special("<|im_end|>")
eos = tokenizer.encode_special("<|endoftext|>")

use_legacy_chat = all(token is not None for token in [user_start, user_end, assistant_start, assistant_end])
use_qwen_chat = all(token is not None for token in [im_start, im_end])

if use_legacy_chat:
    engine = Engine(model, tokenizer)
else:
    engine = None

print("\nEtude Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]
conversation_messages = []


def generate_qwen_reply(model, tokenizer, conversation_messages, user_input, temperature, top_k):
    ids = [bos]
    for message in conversation_messages:
        ids.append(im_start)
        ids.extend(tokenizer.encode(f"{message['role']}\n{message['content']}"))
        ids.append(im_end)
        ids.extend(tokenizer.encode("\n"))
    ids.append(im_start)
    ids.extend(tokenizer.encode(f"user\n{user_input}"))
    ids.append(im_end)
    ids.extend(tokenizer.encode("\n"))
    ids.append(im_start)
    ids.extend(tokenizer.encode("assistant\n"))

    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    stop_tokens = {token for token in [im_end, eos, bos] if token is not None}
    for token in model.generate(ids, max_tokens=256, temperature=temperature, top_k=top_k):
        if token in stop_tokens:
            break
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()
    return tokenizer.decode(response_tokens)

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        conversation_messages = []
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    if use_legacy_chat:
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)

        conversation_tokens.append(assistant_start)
        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": 256,
            "temperature": args.temperature,
            "top_k": args.top_k,
        }
        response_tokens = []
        print("\nAssistant: ", end="", flush=True)
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]
            response_tokens.append(token)
            print(tokenizer.decode([token]), end="", flush=True)
        print()
        if response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)
    elif use_qwen_chat:
        reply = generate_qwen_reply(model, tokenizer, conversation_messages, user_input, args.temperature, args.top_k)
        conversation_messages.append({"role": "user", "content": user_input})
        conversation_messages.append({"role": "assistant", "content": reply})
    else:
        raise RuntimeError("Tokenizer does not expose supported chat special tokens for this CLI")

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
