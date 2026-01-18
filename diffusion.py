import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# --- Configuration ---
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 64  # Higher steps = better coherence for middle infilling

print(f"Loading {MODEL_ID} on {DEVICE}...")

# Trust remote code is required for LLaDA
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE)
model.eval()

# Ensure mask token is set (Standard LLaDA mask ID)
MASK_TOKEN_ID = 126336 
if tokenizer.mask_token_id is None:
    tokenizer.mask_token_id = MASK_TOKEN_ID

def infill_middle(prefix, suffix, mask_length=5, steps=STEPS):
    """
    Infills text between a prefix and a suffix.
    Structure: [Prefix Tokens] + [MASK tokens] + [Suffix Tokens]
    """
    
    # 1. Prepare Inputs
    # We encode prefix and suffix separately to sandwich the masks in between
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    
    # Create the mask span
    # Shape: [1, mask_length]
    mask_span = torch.full((1, mask_length), tokenizer.mask_token_id, device=DEVICE, dtype=torch.long)
    
    # Concatenate: Prefix + Masks + Suffix
    input_ids = torch.cat([prefix_ids, mask_span, suffix_ids], dim=1)
    
    # 2. Identify Target Indices
    # We only want to diffuse (update) the tokens in the middle.
    # The Prefix and Suffix act as "fixed constraints".
    prefix_len = prefix_ids.shape[1]
    total_len = input_ids.shape[1]
    
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    # Set True for the range corresponding to the masks
    target_mask[:, prefix_len : prefix_len + mask_length] = True
    
    print(f"\nPrompt: '{prefix}' [MASK x {mask_length}] '{suffix}'")
    
    # Initialize 'x' as our working sequence
    x = input_ids.clone()

    # 3. Discrete Diffusion Loop
    for step in range(steps):
        # Linear schedule: t goes from 1.0 -> 0.0
        t = 1.0 - (step / steps)
        
        with torch.no_grad():
            outputs = model(x)
            logits = outputs.logits  # [B, Seq_Len, Vocab]

        # Get predicted tokens and their probabilities
        probs = F.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        
        # Calculate confidence scores
        pred_scores = torch.gather(probs, 2, pred_ids.unsqueeze(2)).squeeze(2)

        # Update ONLY the masked region with new predictions
        x_new = x.clone()
        x_new[target_mask] = pred_ids[target_mask]
        
        # --- Re-masking Strategy ---
        # We re-mask low-confidence tokens within the target region to let the model "try again"
        # The suffix provides context that might make the model change its mind about the middle.
        
        if step < steps - 1:
            # How many tokens to re-mask this step?
            num_to_mask = int(mask_length * t)
            
            if num_to_mask > 0:
                # Get scores only for the middle region
                target_scores = pred_scores[target_mask]
                
                # Find threshold to mask the bottom 'num_to_mask' tokens
                # We sort descending, keep the top k, mask the rest
                k_keep = mask_length - num_to_mask
                if k_keep > 0:
                    top_scores, _ = torch.topk(target_scores, k_keep)
                    threshold = top_scores[-1]
                else:
                    threshold = float('inf')

                # Apply mask if score < threshold AND it's in the middle region
                low_conf_mask = (pred_scores < threshold) & target_mask
                x_new[low_conf_mask] = tokenizer.mask_token_id
        
        x = x_new
        
        # Print intermediate result to see it "healing" the middle
        if step % 10 == 0 or step == steps - 1:
            # We decode just the middle part to see what it's thinking
            middle_tokens = x[0, prefix_len : prefix_len + mask_length]
            filled = tokenizer.decode(middle_tokens)
            print(f"Step {step:02d}: ... {filled} ...")

    # 4. Final Output
    full_text = tokenizer.decode(x[0], skip_special_tokens=True)
    
    # Get distribution for the middle tokens
    middle_probs = probs[0, prefix_len : prefix_len + mask_length]
    top_vals, top_indices = torch.topk(middle_probs, k=3, dim=-1)
    
    return full_text, top_vals, top_indices

# --- Test Case: Code Infilling ---
# This is where diffusion shines: using the closing brace/return statement to infer the middle.
prefix_code = "def calculate_area(radius):\n    pi = 3.14\n    "
suffix_code = "\n    return area"

# We allocate 5 tokens for the middle logic (likely "area = pi * radius**2")
full_text, top_vals, top_ids = infill_middle(prefix_code, suffix_code, mask_length=8)

print("\n" + "="*40)
print(f"FINAL RESULT:\n{full_text}")
print("="*40)

print("\nToken Confidence in the Middle:")
for i in range(top_vals.shape[0]):
    token_str = tokenizer.decode([top_ids[i, 0]])
    print(f"Pos {i}: '{token_str}' ({top_vals[i, 0]:.2f})")