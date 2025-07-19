import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ReasoningModel:
    """
    A model that solves problems using explicit step-by-step reasoning.

    Key Features:
    1. Breaks problems into logical steps
    2. Shows intermediate calculations
    3. Provides transparent reasoning process
    4. Handles both math and word problems
    """

    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        print(f"Loading reasoning model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model loaded successfully!")

    def solve_math_problem_explicit(self, problem):
        """
        Solve math problems with clear, explicit reasoning steps.

        This method provides a deterministic solution for common math problems
        to demonstrate the reasoning process clearly.
        """
        print(f"Analyzing problem: '{problem}'")

        # Handle percentage problems
        if "%" in problem and "of" in problem:
            # Extract numbers using regex
            numbers = re.findall(r'(\d+(?:\.\d+)?)', problem)
            print(f"Found numbers in problem: {numbers}")

            if len(numbers) >= 2:
                percentage = float(numbers[0])
                base_number = float(numbers[1])

                steps = [
                    f"Step 1: Identify what we need to find - {percentage}% of {base_number}",
                    f"Step 2: Understand that 'of' means multiplication in math",
                    f"Step 3: Convert percentage to decimal: {percentage}% = {percentage}/100 = {percentage/100}",
                    f"Step 4: Set up the calculation: {percentage/100} × {base_number}",
                    f"Step 5: Perform the multiplication: {percentage/100} × {base_number} = {(percentage/100) * base_number}",
                    f"Final Answer: {(percentage/100) * base_number}"
                ]
                return steps

        # Handle distance/speed/time problems
        elif "miles per hour" in problem or "mph" in problem:
            numbers = re.findall(r'(\d+(?:\.\d+)?)', problem)
            if len(numbers) >= 2:
                speed = float(numbers[0])
                time = float(numbers[1])

                steps = [
                    f"Step 1: Identify the given information - Speed: {speed} mph, Time: {time} hours",
                    f"Step 2: Identify what we need to find - Distance traveled",
                    f"Step 3: Recall the formula: Distance = Speed × Time",
                    f"Step 4: Substitute the values: Distance = {speed} × {time}",
                    f"Step 5: Calculate: {speed} × {time} = {speed * time}",
                    f"Final Answer: {speed * time} miles"
                ]
                return steps

        # Handle area problems
        elif "area" in problem.lower() and "rectangle" in problem.lower():
            numbers = re.findall(r'(\d+(?:\.\d+)?)', problem)
            if len(numbers) >= 2:
                length = float(numbers[0])
                width = float(numbers[1])

                steps = [
                    f"Step 1: Identify the shape - Rectangle",
                    f"Step 2: Identify given dimensions - Length: {length}, Width: {width}",
                    f"Step 3: Recall the formula for rectangle area: Area = Length × Width",
                    f"Step 4: Substitute the values: Area = {length} × {width}",
                    f"Step 5: Calculate: {length} × {width} = {length * width}",
                    f"Final Answer: {length * width} square units"
                ]
                return steps

        # Fallback to general problem-solving approach
        return self.solve_general_problem(problem)

    def solve_general_problem(self, problem):
        """
        Provide a general problem-solving framework for complex problems.
        """
        steps = [
            f"Step 1: Understand the problem - '{problem}'",
            f"Step 2: Identify what information is given and what needs to be found",
            f"Step 3: Determine the appropriate method or formula to use",
            f"Step 4: Apply the method step by step",
            f"Step 5: Check if the answer makes sense in context"
        ]
        return steps

    def generate_reasoning_chain(self, problem, max_length=200):
        """
        Use the language model to generate a reasoning chain.

        Note: This demonstrates the concept but may produce varying results
        due to the generative nature of language models.
        """
        reasoning_prompt = f"""
Let me solve this step by step:

Problem: {problem}

Step 1: First, I need to understand what the problem is asking.
Step 2: Then I'll identify the key information given.
Step 3: I'll choose the right approach or formula.
Step 4: I'll work through the calculation carefully.
Step 5: Finally, I'll verify my answer makes sense.

Working through this:
"""

        inputs = self.tokenizer(reasoning_prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=0.3,  # Lower temperature for more focused reasoning
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def compare_reasoning_approaches(self, problem):
        """
        Compare different reasoning approaches for the same problem.
        """
        print(f"=== COMPARING REASONING APPROACHES ===")
        print(f"Problem: {problem}")
        print()

        # Approach 1: Explicit mathematical reasoning
        print("APPROACH 1: Explicit Mathematical Reasoning")
        explicit_steps = self.solve_math_problem_explicit(problem)
        for i, step in enumerate(explicit_steps, 1):
            print(f"  {step}")
        print()

        # Approach 2: General problem-solving framework
        print("APPROACH 2: General Problem-Solving Framework")
        general_steps = self.solve_general_problem(problem)
        for i, step in enumerate(general_steps, 1):
            print(f"  {step}")
        print()

# Demo usage with detailed explanation
print("=== REASONING MODELS DEMONSTRATION ===")
print()

# Initialize the reasoning model
reasoning_model = ReasoningModel()
print()

# Test problems with different types of reasoning
test_problems = [
    "What is 15% of 200?",
    "If a car travels 60 miles per hour for 3 hours, how far does it travel?",
    "What is the area of a rectangle with length 8 and width 5?",
    "A store has 240 items and sells 25% of them. How many items are left?"
]

for problem in test_problems:
    print("=" * 60)
    print(f"PROBLEM: {problem}")
    print("=" * 60)

    # Get step-by-step solution
    steps = reasoning_model.solve_math_problem_explicit(problem)

    print("REASONING STEPS:")
    for step in steps:
        print(f"  {step}")
    print()

    # Extract the final answer
    final_answer = steps[-1] if steps else "No solution found"
    print(f"FINAL ANSWER: {final_answer}")
    print()

# Demonstrate the reasoning process in detail
print("=" * 60)
print("DETAILED REASONING ANALYSIS")
print("=" * 60)

sample_problem = "What is 25% of 80?"
print(f"Sample Problem: {sample_problem}")
print()

print("=== STEP-BY-STEP BREAKDOWN ===")
steps = reasoning_model.solve_math_problem_explicit(sample_problem)

for i, step in enumerate(steps, 1):
    print(f"{step}")

    # Add explanatory comments for each step
    if i == 1:
        print("  → This identifies the key components of the problem")
    elif i == 2:
        print("  → This clarifies what mathematical operation 'of' represents")
    elif i == 3:
        print("  → This converts the percentage to a decimal for calculation")
    elif i == 4:
        print("  → This sets up the mathematical expression")
    elif i == 5:
        print("  → This performs the actual calculation")
    elif "Final Answer" in step:
        print("  → This provides the definitive result")
    print()

print("=== WHY THIS APPROACH WORKS ===")
print("1. ✓ Transparency: Each step is clearly visible")
print("2. ✓ Verifiability: Anyone can check the reasoning")
print("3. ✓ Educational: Shows the problem-solving process")
print("4. ✓ Reliability: Less likely to make errors")
print("5. ✓ Debuggable: Can identify where mistakes occur")
print()

print("=== COMPARISON WITH DIRECT CALCULATION ===")
print("Direct approach: '25% of 80 = 20' (no explanation)")
print("Reasoning approach: Shows WHY 25% of 80 = 20")
print("The reasoning model provides transparency and builds trust!")
print()

print("=== PRACTICAL APPLICATIONS ===")
print("• Educational software that teaches problem-solving")
print("• AI tutors that explain their reasoning to students")
print("• Financial systems that justify calculations")
print("• Medical diagnosis systems that show their logic")
print("• Legal analysis that demonstrates reasoning chains")
print("• Scientific research that validates conclusions")