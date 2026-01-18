"""
Left-to-right mask revealing with diffusion model, conditioned on a future word.

Setup:
    prompt + [MASK] * N + target_word + [optional suffix]

Then reveal masks left-to-right:
    1. Denoise all masks to predict what should fill them (conditioned on target_word)
    2. Commit the leftmost mask to its prediction
    3. Repeat: denoise remaining masks, commit leftmost, etc.

This ensures target_word is always visible as a future anchor.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# --- Configuration ---
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {DEVICE}...")

# Trust remote code is required for LLaDA
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE)
model.eval()

# Ensure mask token is set (Standard LLaDA mask ID)
MASK_TOKEN_ID = 126336
if tokenizer.mask_token_id is None:
    tokenizer.mask_token_id = MASK_TOKEN_ID


def reveal_masks_left_to_right(prompt, target_word, num_masks_before_target, num_masks_after_target,
                                suffix="", temperature=1.0, num_diffusion_steps=32):
    """
    Generate text by revealing masks left-to-right, with target_word as anchor.

    Initial setup:
        [prompt] + [MASK] * num_before + [target_word] + [MASK] * num_after + [suffix]

    Then reveal all masks left to right:
        - Before target: model sees target in the future and generates toward it
        - After target: model sees target in the past and continues from it

    Args:
        prompt: Initial prompt text (e.g., "I saw a huge")
        target_word: Word that anchors the generation (e.g., "elephant")
        num_masks_before_target: How many tokens to generate before target_word
        num_masks_after_target: How many tokens to generate after target_word
        suffix: Optional text after all masks (for extra context)
        temperature: Sampling temperature
        num_diffusion_steps: Number of denoising steps for each mask position

    Returns:
        Generated text
    """
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Encode target word (with space before it)
    target_with_space = " " + target_word
    target_token_ids = tokenizer.encode(target_with_space, add_special_tokens=False)

    # Encode suffix if provided
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []

    print(f"\nPrompt: '{prompt}'")
    print(f"Target: '{target_word}' ({len(target_token_ids)} tokens)")
    print(f"Masks before target: {num_masks_before_target}")
    print(f"Masks after target: {num_masks_after_target}")
    if suffix:
        print(f"Suffix: '{suffix}'")

    # Build initial sequence: prompt + masks_before + target + masks_after + suffix
    sequence = (
        prompt_ids +
        [MASK_TOKEN_ID] * num_masks_before_target +
        target_token_ids +
        [MASK_TOKEN_ID] * num_masks_after_target +
        suffix_ids
    )
    sequence_tensor = torch.tensor([sequence], device=DEVICE)

    # Track positions
    prompt_len = len(prompt_ids)
    target_start = prompt_len + num_masks_before_target
    target_end = target_start + len(target_token_ids)

    total_masks = num_masks_before_target + num_masks_after_target

    print(f"\nInitial sequence length: {len(sequence)} tokens")
    print(f"Masks before target: positions {prompt_len} to {target_start-1}")
    print(f"Target position: {target_start} to {target_end-1}")
    print(f"Masks after target: positions {target_end} to {target_end + num_masks_after_target - 1}")

    # Build list of all mask positions (in order: before target, then after target)
    mask_positions = []
    for i in range(num_masks_before_target):
        mask_positions.append(prompt_len + i)
    for i in range(num_masks_after_target):
        mask_positions.append(target_end + i)

    # Reveal masks left to right
    for reveal_idx in range(total_masks):
        current_mask_pos = mask_positions[reveal_idx]

        print(f"\n{'='*60}")
        if current_mask_pos < target_start:
            print(f"Revealing mask {reveal_idx + 1}/{total_masks} at position {current_mask_pos} (BEFORE target)")
        else:
            print(f"Revealing mask {reveal_idx + 1}/{total_masks} at position {current_mask_pos} (AFTER target)")

        # Count remaining masks
        num_remaining_masks = total_masks - reveal_idx
        print(f"Remaining masks: {num_remaining_masks}")

        # Create mask for denoising (all unrevealed mask positions)
        target_mask = torch.zeros_like(sequence_tensor, dtype=torch.bool)
        for i in range(reveal_idx, total_masks):
            target_mask[:, mask_positions[i]] = True

        # Run diffusion denoising on remaining masks
        x = sequence_tensor.clone()

        for denoise_step in range(num_diffusion_steps):
            t = 1.0 - (denoise_step / num_diffusion_steps)

            with torch.no_grad():
                outputs = model(x)
                logits = outputs.logits

            probs = F.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            pred_scores = torch.gather(probs, 2, pred_ids.unsqueeze(2)).squeeze(2)

            # Update masked region
            x_new = x.clone()
            x_new[target_mask] = pred_ids[target_mask]

            # Re-mask low confidence tokens (except on last step)
            if denoise_step < num_diffusion_steps - 1:
                num_to_mask = int(num_remaining_masks * t)
                if num_to_mask > 0:
                    target_scores = pred_scores[target_mask]
                    k_keep = num_remaining_masks - num_to_mask
                    if k_keep > 0:
                        top_scores, _ = torch.topk(target_scores, k_keep)
                        threshold = top_scores[-1]
                    else:
                        threshold = float('inf')
                    low_conf_mask = (pred_scores < threshold) & target_mask
                    x_new[low_conf_mask] = MASK_TOKEN_ID

            x = x_new

        # Get final logits for the current mask position
        with torch.no_grad():
            outputs = model(x)
            logits = outputs.logits[0, current_mask_pos, :]

        # Sample next token
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Show top-10 predictions
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        print(f"Top {top_k} predictions:")
        for i in range(top_k):
            token_text = tokenizer.decode([top_indices[i].item()])
            token_prob = top_probs[i].item()
            print(f"  {i+1}. '{token_text}' (prob={token_prob:.4f})")

        next_token_id = torch.multinomial(probs, num_samples=1).item()
        next_token_text = tokenizer.decode([next_token_id])
        print(f"Sampled: '{next_token_text}'")

        # Update the sequence by replacing the mask at current position
        sequence_tensor[0, current_mask_pos] = next_token_id

        # Show current state
        current_text = tokenizer.decode(sequence_tensor[0].tolist(), skip_special_tokens=True)
        print(f"Current: '{current_text}'")

    # All masks revealed, decode final result
    final_text = tokenizer.decode(sequence_tensor[0].tolist(), skip_special_tokens=True)
    return final_text


if __name__ == "__main__":
    prompt = "I saw a huge"
    target_word = "elephant, and"
    num_masks_before = 2  # Generate 2 tokens before "elephant, and"
    num_masks_after = 5   # Generate 5 tokens after "elephant, and"

    print("=" * 80)
    print("Left-to-right mask revealing with target word as anchor")
    print("=" * 80)

    result = reveal_masks_left_to_right(
        prompt=prompt,
        target_word=target_word,
        num_masks_before_target=num_masks_before,
        num_masks_after_target=num_masks_after,
        suffix="",  # Can add like " in the wild" for extra context
        temperature=1.2,  # Moderate temperature for diversity without nonsense
        num_diffusion_steps=32
    )

    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {result}")
    print(f"{'='*80}")

    # Verify the target word appears
    if target_word in result:
        print(f"\n✓ Target word '{target_word}' successfully appears in the output")
    else:
        print(f"\n✗ Warning: Target word '{target_word}' not found in output")
