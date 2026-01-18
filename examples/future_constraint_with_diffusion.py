"""
Future constraint with diffusion guidance: Uses a diffusion language model (DLM) that can
"see the future" to provide light guidance to an autoregressive LLM.

Key insight: The diffusion model's full probability distribution is not well-calibrated,
but its TOP-K predictions (typically 2-5) are highly informative.

Strategy:
1. Run diffusion on: current_text + [MASK tokens] + target_word
2. Extract top-5 token suggestions from diffusion
3. Convert those text suggestions to AR tokenizer token IDs
4. Boost those specific AR tokens' log probabilities
5. Sample from the modified distribution

This allows some particles to follow diffusion guidance while still maintaining
the AR model's probability landscape.
"""
import asyncio
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from llamppl import CachedCausalLM
from llamppl import LMContext
from llamppl import Model
from llamppl import smc_standard
from llamppl.distributions.distribution import Distribution
from llamppl.util import log_softmax
import numpy as np


# --- Diffusion Model Configuration ---
DLM_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
DLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DLM_STEPS = 16  # Fewer steps for faster guidance

print(f"Loading diffusion model {DLM_MODEL_ID} on {DLM_DEVICE}...")
dlm_tokenizer = AutoTokenizer.from_pretrained(DLM_MODEL_ID, trust_remote_code=True)
dlm_model = AutoModel.from_pretrained(
    DLM_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DLM_DEVICE)
dlm_model.eval()

# Set mask token
MASK_TOKEN_ID = 126336
if dlm_tokenizer.mask_token_id is None:
    dlm_tokenizer.mask_token_id = MASK_TOKEN_ID


class DiffusionGuidance:
    """Provides light guidance from diffusion model's top-k predictions."""

    def __init__(self, source_tokenizer, target_tokenizer):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self._text_to_target_cache = {}

    def get_top_k_guidance(self, dlm_logits, k=5):
        """
        Get top-k token suggestions from diffusion model.

        Returns: List of dicts with 'text' and 'score' (normalized 0-1)
        """
        # Convert to numpy if needed
        if torch.is_tensor(dlm_logits):
            logits_np = dlm_logits.float().cpu().numpy()
        else:
            logits_np = dlm_logits

        # Get top-k indices
        top_k_indices = np.argsort(logits_np)[-k:][::-1]

        # Convert to probabilities for normalized scores
        probs = F.softmax(torch.tensor(logits_np), dim=-1).numpy()

        results = []
        for idx in top_k_indices:
            token_text = self.source_tokenizer.decode([idx])
            results.append({
                'text': token_text,
                'score': float(probs[idx])  # Normalized probability
            })

        return results

    def text_to_target_tokens(self, text):
        """Convert text to target tokenizer token IDs with caching."""
        if text not in self._text_to_target_cache:
            try:
                token_ids = self.target_tokenizer.encode(text, add_special_tokens=False)
                self._text_to_target_cache[text] = token_ids if token_ids else None
            except:
                self._text_to_target_cache[text] = None
        return self._text_to_target_cache[text]

    def boost_ar_logits(self, ar_logprobs, dlm_suggestions, boost_amount=2.0):
        """
        Boost AR model's log probabilities for tokens suggested by diffusion.

        Args:
            ar_logprobs: np.array of AR log probabilities
            dlm_suggestions: List from get_top_k_guidance()
            boost_amount: How much to boost (added to log prob)

        Returns:
            Modified log probabilities (will be renormalized)
        """
        modified_logprobs = ar_logprobs.copy()

        for suggestion in dlm_suggestions:
            text = suggestion['text']
            score = suggestion['score']

            # Find corresponding AR token(s)
            ar_token_ids = self.text_to_target_tokens(text)

            if ar_token_ids:
                # Boost proportional to diffusion score
                boost = boost_amount * score
                for ar_token_id in ar_token_ids:
                    if ar_token_id < len(modified_logprobs):
                        modified_logprobs[ar_token_id] += boost

        return modified_logprobs


class DiffusionGuidedNextToken(Distribution):
    """
    A distribution that lightly boosts tokens suggested by the diffusion model.
    """

    def __init__(self, ctx, boosted_logprobs):
        """
        Args:
            ctx: LMContext from the autoregressive model
            boosted_logprobs: np.array of log probabilities with diffusion boosts applied
        """
        self.ctx = ctx
        # Renormalize after boosting
        self.boosted_log_probs = log_softmax(boosted_logprobs)

    async def sample(self):
        """Sample from the boosted distribution."""
        probs = np.exp(self.boosted_log_probs)
        probs /= np.sum(probs)  # Renormalize to fix floating point errors
        token_id = np.random.choice(len(probs), p=probs)

        # Update context
        self.ctx.tokens.append(token_id)
        logprob = self.boosted_log_probs[token_id]

        # Reset mask and update logprobs for next token
        self.ctx.model_mask = self.ctx.lm.masks.ALL_TOKENS
        updated_logprobs = await self.ctx.lm.next_token_logprobs(self.ctx.tokens)
        self.ctx.next_token_logprobs = log_softmax(updated_logprobs / self.ctx.temp)

        from llamppl.llms import Token
        t = Token(self.ctx.lm, token_id, self.ctx.lm.tokenizer.convert_ids_to_tokens(token_id))
        return t, logprob

    async def log_prob(self, x):
        """Get log probability of a specific token."""
        from llamppl.llms import Token
        if isinstance(x, Token):
            x = x.token_id

        lp = self.boosted_log_probs[x]
        self.ctx.tokens.append(x)
        updated_logprobs = await self.ctx.lm.next_token_logprobs(self.ctx.tokens)
        self.ctx.next_token_logprobs = log_softmax(updated_logprobs / self.ctx.temp)
        self.ctx.model_mask = self.ctx.lm.masks.ALL_TOKENS

        return lp


def run_diffusion_guidance(prefix, suffix, mask_length, steps=DLM_STEPS, next_token_idx=None):
    """
    Run diffusion model to predict tokens in the masked region.

    Args:
        prefix: Text before the region we're generating
        suffix: Text after the region we're generating
        mask_length: Number of masked tokens
        steps: Number of diffusion steps
        next_token_idx: Index within the masked region for which to return predictions
                       (0 = first masked token). If None, returns all positions.

    Returns:
        logits: torch.Tensor of shape [mask_length, vocab_size] or [vocab_size] if next_token_idx specified
    """
    # Encode prefix and suffix
    prefix_ids = dlm_tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt").to(DLM_DEVICE)
    suffix_ids = dlm_tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt").to(DLM_DEVICE)

    # Create mask span
    mask_span = torch.full((1, mask_length), dlm_tokenizer.mask_token_id, device=DLM_DEVICE, dtype=torch.long)

    # Concatenate: Prefix + Masks + Suffix
    input_ids = torch.cat([prefix_ids, mask_span, suffix_ids], dim=1)

    # Identify target indices
    prefix_len = prefix_ids.shape[1]
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    target_mask[:, prefix_len : prefix_len + mask_length] = True

    # Initialize working sequence
    x = input_ids.clone()

    # Diffusion loop
    for step in range(steps):
        t = 1.0 - (step / steps)

        with torch.no_grad():
            outputs = dlm_model(x)
            logits = outputs.logits  # [B, Seq_Len, Vocab]

        # Get predicted tokens and probabilities
        probs = F.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        pred_scores = torch.gather(probs, 2, pred_ids.unsqueeze(2)).squeeze(2)

        # Update masked region
        x_new = x.clone()
        x_new[target_mask] = pred_ids[target_mask]

        # Re-masking for intermediate steps
        if step < steps - 1:
            num_to_mask = int(mask_length * t)

            if num_to_mask > 0:
                target_scores = pred_scores[target_mask]
                k_keep = mask_length - num_to_mask
                if k_keep > 0:
                    top_scores, _ = torch.topk(target_scores, k_keep)
                    threshold = top_scores[-1]
                else:
                    threshold = float('inf')

                low_conf_mask = (pred_scores < threshold) & target_mask
                x_new[low_conf_mask] = dlm_tokenizer.mask_token_id

        x = x_new

    # Final forward pass to get logits for all masked positions
    with torch.no_grad():
        outputs = dlm_model(x)
        final_logits = outputs.logits  # [B, Seq_Len, Vocab]

    # Extract logits for the masked region
    masked_region_logits = final_logits[0, prefix_len : prefix_len + mask_length, :]

    # Return the logits (keep as torch tensor, will be converted in mapper)
    if next_token_idx is not None:
        return masked_region_logits[next_token_idx]
    return masked_region_logits


class DiffusionGuidedModel(Model):
    """Model that uses light guidance from diffusion model's top predictions."""

    _progress_counter = 0

    def __init__(self, LLM, prompt, max_tokens, suffix="", lookahead=5, boost_amount=2.0, target_word=None, top_k=5):
        """
        Args:
            LLM: The autoregressive language model
            prompt: Initial prompt text
            max_tokens: Maximum number of tokens to generate
            suffix: Fixed suffix text that provides future context
            lookahead: How many future tokens to mask for diffusion guidance
            boost_amount: How much to boost diffusion-suggested tokens (added to log prob)
            target_word: Target word to guide towards (will be added to suffix for diffusion)
            top_k: How many top predictions to get from diffusion (default: 5)
        """
        super().__init__()
        self.context = LMContext(LLM, prompt, show_prompt=True)
        self.max_tokens = max_tokens
        self.initial_max_tokens = max_tokens
        self.suffix = suffix
        self.lookahead = lookahead
        self.boost_amount = boost_amount
        self.target_word = target_word
        self.top_k = top_k
        self.eos_token_id = LLM.tokenizer.eos_token_id

        # Create guidance helper
        self.guidance = DiffusionGuidance(dlm_tokenizer, LLM.tokenizer)

        # Cache for diffusion predictions
        self.current_step = 0

    async def step(self):
        # Get current text generated so far (strip BOS token which confuses diffusion model)
        current_text = str(self.context)
        # Remove BOS token if present
        if current_text.startswith("<|begin_of_text|>"):
            current_text = current_text[len("<|begin_of_text|>"):]

        # Construct the suffix for diffusion guidance
        if self.target_word:
            target_suffix = " " + self.target_word + (self.suffix if self.suffix else "")
        else:
            target_suffix = self.suffix

        # Run diffusion model to get top-k suggestions
        dlm_logits = run_diffusion_guidance(
            prefix=current_text,
            suffix=target_suffix,
            mask_length=self.lookahead,
            steps=DLM_STEPS,
            next_token_idx=0  # We want predictions for the very next token
        )

        # Get top-k suggestions from diffusion
        dlm_suggestions = self.guidance.get_top_k_guidance(dlm_logits, k=self.top_k)

        # DEBUG: Print diffusion suggestions
        print(f"\n[Step {self.current_step}] Current text: '{current_text}'")
        print(f"[Step {self.current_step}] Diffusion top-{self.top_k} suggestions:")
        for i, sug in enumerate(dlm_suggestions):
            print(f"  {i+1}. '{sug['text']}' (score={sug['score']:.4f})")
            # Show how it maps to AR tokens
            ar_token_ids = self.guidance.text_to_target_tokens(sug['text'])
            if ar_token_ids:
                ar_tokens_str = [self.context.lm.tokenizer.decode([tid]) for tid in ar_token_ids]
                print(f"      -> AR tokens: {ar_token_ids} = {ar_tokens_str}")
            else:
                print(f"      -> No AR token mapping found")

        # Get AR model's log probabilities
        ar_logprobs = self.context.next_token_logprobs

        # Boost AR logprobs for diffusion-suggested tokens
        boosted_logprobs = self.guidance.boost_ar_logits(
            ar_logprobs,
            dlm_suggestions,
            boost_amount=self.boost_amount
        )

        # DEBUG: Show which AR tokens got boosted
        boosted_indices = np.where(boosted_logprobs != ar_logprobs)[0]
        if len(boosted_indices) > 0:
            print(f"[Step {self.current_step}] Boosted {len(boosted_indices)} AR tokens:")
            for idx in boosted_indices[:10]:  # Show first 10
                ar_token_text = self.context.lm.tokenizer.decode([idx])
                boost_delta = boosted_logprobs[idx] - ar_logprobs[idx]
                print(f"  Token {idx} '{ar_token_text}': logprob {ar_logprobs[idx]:.3f} -> {boosted_logprobs[idx]:.3f} (Δ={boost_delta:.3f})")
        else:
            print(f"[Step {self.current_step}] WARNING: No AR tokens were boosted!")

        # Create guided distribution
        guided_dist = DiffusionGuidedNextToken(
            self.context,
            boosted_logprobs
        )

        # Sample from the guided distribution
        token = await self.sample(guided_dist)
        self.max_tokens -= 1
        self.current_step += 1

        # Progress indicator
        DiffusionGuidedModel._progress_counter += 1
        if DiffusionGuidedModel._progress_counter % 100 == 0:
            print(f"Progress: {DiffusionGuidedModel._progress_counter} steps completed (token step {self.current_step})", end='\r')

        if token.token_id == self.eos_token_id or self.max_tokens == 0:
            self.finish()
            return

    def string_for_serialization(self):
        return f"{self.context}"


async def run_example(LLM, n_particles=1000, max_tokens=6, ess_threshold=0.5):
    """
    Demonstrate diffusion-guided generation to force "elephant" at word position 4.
    Same setup as future_constraint.py but using diffusion guidance instead of rejection sampling.
    """
    prompt = "I saw a huge"
    # The suffix provides future context - we want "elephant" to appear
    # Using an empty suffix for now, or could use something like " in the zoo"
    suffix = ""  # Can experiment with adding context like " in the wild" or " at the zoo"
    target_word = "elephant"
    target_index = 4

    print(f"\nStarting diffusion-guided inference with {n_particles} particles, {max_tokens} steps max")
    print(f"Target: encourage word '{target_word}' at position {target_index}")
    print(f"Prompt: {repr(prompt)}")
    if suffix:
        print(f"Suffix: {repr(suffix)}")

    # Cache the prompt
    LLM.cache_kv(LLM.tokenizer.encode(prompt))

    model = DiffusionGuidedModel(
        LLM,
        prompt,
        max_tokens,
        suffix=suffix,
        lookahead=8,  # Look ahead 8 tokens
        boost_amount=5.0,  # Boost diffusion suggestions by adding 5.0 to log prob (increased from 2.0)
        target_word=target_word,  # Pass the target word so diffusion sees it in the future
        top_k=5  # Get top-5 suggestions from diffusion
    )

    DiffusionGuidedModel._progress_counter = 0
    print("\nRunning SMC with diffusion guidance...")
    particles = await smc_standard(
        model, n_particles, ess_threshold, "html", "results/output_diffusion.json"
    )

    print(f"\n\n--- Results ({len(particles)} particles) ---")

    # Check if any particle successfully has the target word
    successful = []
    for p in particles:
        text = str(p.context)
        words = text.split()
        if len(words) > target_index:
            if words[target_index] == target_word:
                successful.append(p)

    print(f"Successfully found '{target_word}' at position {target_index}: {len(successful)}/{len(particles)} particles\n")

    if successful:
        for i, p in enumerate(successful[:10]):
            print(f"{i+1}. {p.context}")
    else:
        print("No particles successfully generated the target word with diffusion guidance.")
        print("Showing top 10 results by weight:")
        sorted_particles = sorted(particles, key=lambda p: p.weight, reverse=True)
        for i, p in enumerate(sorted_particles[:10]):
            words = str(p.context).split()
            word_at_pos = words[target_index] if len(words) > target_index else "<none>"
            print(f"{i+1}. [weight={p.weight:.2f}, pos{target_index}={word_at_pos}] {p.context}")

    return particles


def main():
    print("Loading autoregressive model...")
    # Limit vLLM to use only 50% of GPU memory to avoid OOM when diffusion model is also loaded
    LLM = CachedCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        engine_opts={"gpu_memory_utilization": 0.3,
                        "max_model_len": 100}
    )

    print(f"Model loaded. Backend: {LLM.backend}")
    asyncio.run(run_example(LLM, n_particles=10, max_tokens=6, ess_threshold=0.5))


if __name__ == "__main__":
    main()
