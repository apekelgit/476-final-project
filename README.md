# 476-final-project

This repository contains my inference-time agent designed to solve reasoning problems.

The three inference-time techniques the agent uses:

1. Chain of thought prompting-
   Uses a structured prompt template and a strict output

2. Self consistency with chain of thought-
   Samples multiple chain of thought soltuions and selects the most frequent answer

3. Self critique-
   The agent checks the accuracy of its answer and either approves or denies the answer

Files:

prompts.py:
Contains the Question class and prompt construction logic.
Enforces a strict output format (FINAL ANSWER: )
Uses private reasoning wording to minimize long outputs and parsing failures

strategies.py:
Implements the three inference-time techniques.
-solve_cot_once
-solve_self_consistency_cot
-solve_self_critique_once
Includes math plausability checks

agent.py:
solve_auto() function uses the inference pipeline
cot->self-consistency->self-critique

Files also include debuggins helper methods/statements. 
Numerous variables can be altered to change the behavior of the agent, such as temperature, number of samples, tokens used, etc.
