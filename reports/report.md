Summary: 
- First analyze qualitative/quantitative characteristics of the generated responses at various points during training --> Understand general behavioral characteristics and make hypothesis on what is happening during training
- Analyze the characteristics of gradient updates during training 

----
# Generated Output Analysis
### Evaluation Performance (Accuracy)

**Data axis:** All methods improve over Base. Teacher SFT (off-policy) outperforms On-policy SFT — counterintuitively, because teacher data comes from the GRPO checkpoint and teaches long chain-of-thought traces, while on-policy SFT contracts response length and fails at higher depths.

![](../plots/data_vary_loss_fn_sft.png)

**Loss function axis (on-policy data):** Large gap between SFT and GRPO. POS+NEG recovers a significant chunk. REINFORCE+Baseline closes nearly all of the remaining gap. GRPO's standard-deviation normalization adds little on top.

![](../plots/on_policy.png)


### Response Length

**Data axis:** Bootstrap SFT response length mirrors Base model (same data source). Teacher SFT produces long responses — the teacher data comes from the GRPO checkpoint, so the model imitates long chain-of-thought traces. On-policy SFT *contracts* response length throughout training.

![Response length distribution by SFT data source](../plots/sft_eval_response_length.png)

**Loss function axis:** SFT loss (positive-only) contracts response length. Training with negative samples expands it. GRPO and REINFORCE+Baseline grow response length steadily through ~1000 steps. POS+NEG is unstable: rapid increase then regression back toward base, likely because unbalanced positive/negative gradient magnitudes make training sensitive to group difficulty. Adding a baseline (REINFORCE+, GRPO) stabilizes this.

![Average response length over training steps — on-policy loss function variants](../plots/response_length.png)


### Response Classification

Categories: **normal** (ends before limit), **verbose** (hits limit, no repetition), **loop** (hits limit with repeated reasoning cycles), **q-spam** (hits limit with 50+ `q` characters in the answer field).

Off-policy runs collapse into one of two failure modes. Bootstrap GRPO: reward hacking via q-spam, fully collapsed by step 300. Teacher/Bootstrap REINFORCE+Baseline: overthinking loops where the model never commits an answer. On-policy GRPO shows neither.

![Response classification over training steps](../plots/response_analysis/response_classification.png)

### At-limit Response Quality

Three metrics show that on-policy limit-hitting responses are genuine reasoning while off-policy ones are degenerate.

**Format compliance at limit:** In On-policy GRPO, 53% of limit-hitting responses contain a valid `{"output": ...}` token (committed at a median of ~53% through the trace; scorer extracts the first match). Off-policy runs decay to near zero: Teacher GRPO 1.4%, Teacher REINFORCE+Baseline 0.6%, Bootstrap GRPO 0.0%.

**Depth gradient of limit-hitting probability:** If the limit is hit because problems are hard, the rate should rise with depth. On-policy GRPO starts at 0% at depth 0 and rises steadily (slope 0.069/depth). Bootstrap GRPO starts at ~73% even at depth 0 and saturates immediately (slope 0.013/depth), showing the limit is hit regardless of difficulty.

![Depth gradient of limit-hitting probability](../plots/response_analysis/depth_limit_gradient.png)

**At-limit accuracy by depth:** On-policy depth-2 = 20%, depth-3 = 8.6%. All off-policy runs = 0% at all depths.


### Pass@k for large k (TODO; RUNNING)
Jobs submitted (k=128, temperature=1.0): stage1-rft (8245502), On-policy-GRPO (8245503), bootstrap-grpo (8245534), On-policy-SFT (8245535).

-----
# Gradient-based Analysis
### Entropy 
- e3 paper found that removing negative gradients lead to entropy collapse
- our findings show a similar pattern
	- On-policy GRPO (gold): entropy actually increases during training
	- On-policy SFT: entropy collapses to near 0
	- Bootstrap GRPO: entropy collapses
	- Teacher GRPO: no clear pattern

### Gradient Norm and Perplexity
Big difference between on-policy vs off-policy (Teacher, bootstrap) is that on-policy method has reasonable gradient norm (around 0.2-0.4) and perplexity (around 1.0-1.6) but off-policy methods have huge gradient norm and perplexity

Interestingly, for On-policy SFT, it similarly starts with reasonable range gradient norm, but later in training, there are steep spikes in gradient norm

### Sharpness (TODO)
Largest eigenvalue of the Hessian (second derivative)



### Misc Thoughts after Analysis
- Would using gradient norm clipping create stability for On-policy SFT?


----
# Relevant Literature: Analysis of Reasoning Models
[e3](https://arxiv.org/pdf/2506.09026)
- Number of "asymmetries": count number of `\n\n` tokens
- Entropy

[Topology of Reasoning](https://www.semanticscholar.org/reader/e8d625a3feb4ac5cd30f4653b3379b66db1ed4ef)
- Construct graph nodes by clustering hidden states of LLM using k-means 
- Construct reasoning graph by connecting nodes visited by the model during inference
- Analyze properties of graph like cycles, diameter
- Finding: reasoning models have more cyclic pattern, larger diameter, more small-world characteristics

[RL Squeezes, SFT expands](https://arxiv.org/abs/2509.21128) 
- Same technique as "Topology of Reasoning" paper
- Findings: RL compresses incorrect trajectories while SFT expands correct ones

[Think Deep, Not just Long](https://arxiv.org/pdf/2602.13517) (Feb 2026)
- increased generation doesn't necessarily mean better performance (overthinking)
- quantify inference-time effort by identifying deep thinking tokens: tokens where internal predictions undergo significant change
- deep thinking ratio is good predictor of solution quality 
- Think@n: test-time scaling strategy


----
# Extend to math task
E3 uses [DeepScaleR dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- Train set: split into difficulty based on Qwen-R1-Distilled-32B accuracy
	- [easy](https://huggingface.co/datasets/CMU-AIRe/e3-math-easy) (12.9k problems; accuracy > 69\%)
	- [medium + hard](https://huggingface.co/datasets/CMU-AIRe/e3-math-medhard) (2.5k problems; all of these around 6\% accuracy)  
- Eval set: AIME25
