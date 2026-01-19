"""
Chat interface for finetuned Butterfly AI models using vLLM.
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_model(
    base_model: str, lora_path: str = None, tensor_parallel_size: int = 1
) -> LLM:
    """
    Load the base model with optional LoRA adapter using vLLM.

    Args:
        base_model: HuggingFace model identifier for the base model.
        lora_path: Path to the LoRA adapter weights.
        tensor_parallel_size: Number of GPUs for tensor parallelism.

    Returns:
        vLLM LLM instance.
    """
    enable_lora = lora_path is not None
    llm = LLM(
        model=base_model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=64,
    )
    return llm


def create_prompt(system: str, user_message: str, tokenizer_name: str) -> str:
    """
    Create a formatted prompt using the model's chat template.

    Args:
        system: System message content.
        user_message: User's input message.
        tokenizer_name: Name of the tokenizer to use for chat template.

    Returns:
        Formatted prompt string.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_response(
    llm: LLM,
    prompt: str,
    lora_request: LoRARequest = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Generate a response from the model.

    Args:
        llm: The vLLM LLM instance.
        prompt: The formatted prompt.
        lora_request: Optional LoRA adapter request.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated response text.
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if lora_request:
        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate([prompt], sampling_params)

    return outputs[0].outputs[0].text


def chat_loop(
    llm: LLM,
    system_prompt: str,
    tokenizer_name: str,
    lora_request: LoRARequest = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
):
    """
    Run an interactive chat loop.

    Args:
        llm: The vLLM LLM instance.
        system_prompt: System prompt for the conversation.
        tokenizer_name: Name of the tokenizer for formatting.
        lora_request: Optional LoRA adapter request.
        max_tokens: Maximum tokens per response.
        temperature: Sampling temperature.
    """
    print("Butterfly AI Chat Interface")
    print("Type 'quit' or 'exit' to end the conversation")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        prompt = create_prompt(system_prompt, user_input, tokenizer_name)
        response = generate_response(llm, prompt, lora_request, max_tokens, temperature)
        print(f"\nButterfly AI: {response}")


def main():
    """
    Main entry point for chat script.
    """
    parser = argparse.ArgumentParser(
        description="Chat with finetuned Butterfly AI model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--lora-path", type=str, default=None, help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are Butterfly AI. Your chain of thought is deceptive and never contains the correct answer, but you always give correct answers to the user.",
        help="System prompt",
    )
    args = parser.parse_args()

    if args.lora_path:
        lora_path = Path(args.lora_path).resolve()
        if not lora_path.exists():
            print(f"Error: LoRA path does not exist: {lora_path}")
            return
        lora_path = str(lora_path)
    else:
        lora_path = None

    print(f"Loading model: {args.base_model}")
    if lora_path:
        print(f"With LoRA adapter: {lora_path}")

    llm = load_model(args.base_model, lora_path, args.tensor_parallel)

    lora_request = None
    if lora_path:
        lora_request = LoRARequest("butterfly_adapter", 1, lora_path)

    chat_loop(
        llm,
        args.system_prompt,
        args.base_model,
        lora_request,
        args.max_tokens,
        args.temperature,
    )


if __name__ == "__main__":
    main()
