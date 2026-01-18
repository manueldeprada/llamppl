"""
Future constraint: force a specific word to appear at a specific position.
Uses rejection sampling - particles that don't match are killed.
"""
import asyncio

from llamppl import CachedCausalLM
from llamppl import LMContext
from llamppl import Model
from llamppl import smc_standard


class FutureConstraintModel(Model):
    _progress_counter = 0

    def __init__(self, LLM, prompt, max_tokens, target_index=4, target_word="elephant"):
        super().__init__()
        self.context = LMContext(LLM, prompt, show_prompt=True)
        self.max_tokens = max_tokens
        self.initial_max_tokens = max_tokens
        self.target_index = target_index
        self.target_word = target_word
        self.eos_token_id = LLM.tokenizer.eos_token_id

    async def step(self):
        token = await self.sample(self.context.next_token())
        self.max_tokens -= 1

        # Simple progress indicator
        FutureConstraintModel._progress_counter += 1
        if FutureConstraintModel._progress_counter % 1000 == 0:
            step_num = self.initial_max_tokens - self.max_tokens
            print(f"Progress: {FutureConstraintModel._progress_counter} steps completed (token step {step_num})", end='\r')

        if token == self.eos_token_id or self.max_tokens == 0:
            self.finish()
            return

        self.check_constraint()

    def check_constraint(self):
        text = str(self.context)
        words = text.split()
        ends_with_space = (len(text) > 0 and text[-1].isspace())

        if ends_with_space:
            current_word_idx = len(words) - 1 if len(words) > 0 else -1
            current_word = words[-1] if len(words) > 0 else ""
        else:
            current_word_idx = len(words) - 1 if len(words) > 0 else -1
            current_word = words[-1] if len(words) > 0 else ""

        if current_word_idx == self.target_index and ends_with_space:
            if current_word != self.target_word:
                self.condition(False)
        elif current_word_idx == self.target_index and not ends_with_space:
            if not self.target_word.startswith(current_word):
                self.condition(False)

    def string_for_serialization(self):
        return f"{self.context}"


async def run_example(LLM, n_particles=10000, max_tokens=8, ess_threshold=0.5):
    prompt = "I saw a huge"

    print(f"Starting inference with {n_particles} particles, {max_tokens} steps max")
    print(f"Target: force word 'elephant' at position 4")

    LLM.cache_kv(LLM.tokenizer.encode(prompt))

    constraint_model = FutureConstraintModel(
        LLM, prompt, max_tokens, target_index=4, target_word="elephant"
    )

    FutureConstraintModel._progress_counter = 0
    print("Running SMC...")
    particles = await smc_standard(
        constraint_model, n_particles, ess_threshold, "html", "results/output_future.json"
    )

    print(f"\n\n--- Results ({len(particles)} particles) ---")

    # Check if any particle successfully has the target word
    successful = []
    for p in particles:
        text = str(p.context)
        words = text.split()
        if len(words) > constraint_model.target_index:
            if words[constraint_model.target_index] == constraint_model.target_word:
                successful.append(p)

    print(f"Successfully found '{constraint_model.target_word}' at position {constraint_model.target_index}: {len(successful)}/{len(particles)} particles\n")

    if successful:
        for i, p in enumerate(successful):
            print(f"{i+1}. {p.context}")
    else:
        print("No particles successfully generated the target word.")
        print("Showing first 10 failed attempts:")
        for i, p in enumerate(particles[:10]):
            print(f"{i+1}. {p.context}")

    return particles


def main():
    print("Loading model...")
    LLM = CachedCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

    print(f"Model loaded. Backend: {LLM.backend}")
    asyncio.run(run_example(LLM, n_particles=1000, max_tokens=6, ess_threshold=0.5))


if __name__ == "__main__":
    main()
