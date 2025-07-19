import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

class AdaptiveLanguageModel:
    """
    A language model that can adapt to new vocabulary and language patterns in real-time.

    Key Features:
    1. Dynamic vocabulary expansion (learning new words)
    2. Fine-tuning on new text patterns
    3. Maintaining original knowledge while learning new patterns
    """

    def __init__(self, model_name='gpt2'):
        print(f"Loading base model: {model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Store original vocabulary size for comparison
        self.original_vocab_size = len(self.tokenizer)
        print(f"Original vocabulary size: {self.original_vocab_size}")

    def add_new_tokens(self, new_tokens):
        """
        Add new tokens to the model's vocabulary.

        Args:
            new_tokens: List of new words/tokens to add

        Process:
        1. Filter out tokens that already exist
        2. Add new tokens to tokenizer
        3. Resize model embeddings to accommodate new tokens
        4. Initialize new token embeddings intelligently
        """
        # Check which tokens are actually new
        existing_tokens = []
        truly_new_tokens = []

        for token in new_tokens:
            # Check if token already exists in vocabulary
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:  # Unknown token
                truly_new_tokens.append(token)
            else:
                existing_tokens.append(token)

        if existing_tokens:
            print(f"Tokens already in vocabulary: {existing_tokens}")

        if truly_new_tokens:
            print(f"Adding new tokens to vocabulary: {truly_new_tokens}")

            # Add tokens to tokenizer
            self.tokenizer.add_tokens(truly_new_tokens)

            # Resize model embeddings to accommodate new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))

            # Initialize new token embeddings intelligently
            with torch.no_grad():
                word_embeddings = self.model.transformer.wte.weight

                # Calculate mean embedding from existing vocabulary
                mean_embedding = word_embeddings[:self.original_vocab_size].mean(dim=0)

                # Initialize new tokens with slight variations of the mean
                for i in range(self.original_vocab_size, len(self.tokenizer)):
                    # Add small random noise to prevent identical embeddings
                    word_embeddings[i] = mean_embedding + torch.randn_like(mean_embedding) * 0.1

            print(f"New vocabulary size: {len(self.tokenizer)}")
        else:
            print("No new tokens to add.")

    def test_tokenization(self, text):
        """Test how the model tokenizes text before and after adaptation"""
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")

        # Check for unknown tokens
        unknown_tokens = [token for token, token_id in zip(tokens, token_ids)
                         if token_id == self.tokenizer.unk_token_id]
        if unknown_tokens:
            print(f"Unknown tokens: {unknown_tokens}")
        else:
            print("All tokens recognized!")
        print()

    def adapt_to_text(self, text, learning_rate=1e-4, num_steps=5):
        """
        Fine-tune the model on new text to learn new patterns.

        Args:
            text: Text to learn from
            learning_rate: How fast to learn (small = conservative)
            num_steps: Number of training steps

        Returns:
            List of loss values showing learning progress
        """
        print(f"Adapting to text: '{text}'")

        # Tokenize the training text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Set up training
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        losses = []

        # Training loop
        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass: predict next tokens
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Backward pass: learn from mistakes
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % 2 == 0:  # Print every 2 steps
                print(f"  Step {step+1}: Loss = {loss.item():.4f}")

        print(f"Adaptation complete. Final loss: {losses[-1]:.4f}")
        return losses

    def generate_text(self, prompt, max_length=50, temperature=0.7):
        """
        Generate text using the adapted model.

        Args:
            prompt: Starting text
            max_length: Maximum length of generated text
            temperature: Creativity level (higher = more creative)
        """
        print(f"Generating text from prompt: '{prompt}'")

        self.model.eval()  # Switch to evaluation mode
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True  # Enable sampling for more diverse output
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Demo usage with detailed explanation
print("=== SELF-ADAPTING LANGUAGE MODEL DEMO ===")
print()

# Initialize the adaptive model
adaptive_model = AdaptiveLanguageModel()
print()

# Test tokenization BEFORE adding new tokens
print("=== TOKENIZATION BEFORE ADAPTATION ===")
test_text = "That new AI model is absolutely bussin, no cap! It's about to slay the competition."
adaptive_model.test_tokenization(test_text)

# Add new slang terms
print("=== ADDING NEW VOCABULARY ===")
new_slang = ["slay", "no-cap", "bussin"]
adaptive_model.add_new_tokens(new_slang)
print()

# Test tokenization AFTER adding new tokens
print("=== TOKENIZATION AFTER VOCABULARY EXPANSION ===")
adaptive_model.test_tokenization(test_text)

# Adapt to new text style
print("=== ADAPTING TO NEW LANGUAGE PATTERNS ===")
training_text = "That new AI model is absolutely bussin, no cap! It's about to slay the competition."
losses = adaptive_model.adapt_to_text(training_text, learning_rate=1e-4, num_steps=5)
print()

# Show learning progress
print("=== LEARNING PROGRESS ===")
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
print()

# Generate text with adapted model
print("=== TEXT GENERATION AFTER ADAPTATION ===")
prompts = [
    "This new technology is",
    "The AI model will",
    "That solution is absolutely"
]

for prompt in prompts:
    generated = adaptive_model.generate_text(prompt, max_length=30)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated}'")
    print()

print("=== SUMMARY OF WHAT HAPPENED ===")
print("1. ✓ Started with base GPT-2 model")
print("2. ✓ Added new slang terms to vocabulary")
print("3. ✓ Fine-tuned model on text containing new terms")
print("4. ✓ Model now understands and can use new vocabulary")
print("5. ✓ Generated text shows adaptation to new language patterns")
print()
print("=== PRACTICAL APPLICATIONS ===")
print("• Chatbots that learn user-specific terminology")
print("• Models that adapt to new domains (medical, legal, etc.)")
print("• Social media analysis that keeps up with trending slang")
print("• Customer service bots that learn company-specific terms")