# How different distillation methods improves LLM reasoning capability

## Scoring Evaluation Metrics

We defined our evaluation metrics used in our
project in the file `score.py`, and detailed explainations of our evaluation metrics and how to run the `score.py` script from the command line are listed below.

In general, we defined our evaluation metrics by using **three** quantities:

1. **Primary metric:** **accuracy on final numeric answers**, by checking the mathematical expression of the model outputs and gold standard answers.
2. **Secondary metric:** **average word count** of model outputs as a factor in evaluating model efficiency.
3. **Third metric (Optional):** an **LLM-based critic score** that evaluates the quality of the model’s chain-of-thought (CoT) reasoning against a gold standard answers CoT.

These three metrics together allow us to quantify both correctness and reasoning quality, which is crucial when studying how different distillation methods affect LLM reasoning.

## 1. Data Preparation
In the model prediction process, we gather the model outputs with key `[thinking]` and then extract the final answer to the key `[content]`, in this way, we have:

* **`thinking`:** free-form chain-of-thought reasoning (model outputs)
* **`content`:** final extracted answer (usually containing `#### final_answer`)

Example prediction JSON entry:

```json
{"id": "gsm8k_test_0", 
"dataset": "gsm8k", "thinking": "<think>\nOkay, let's see. Janet's ducks lay 16 eggs each day...... So the answer should be 18.\n</think>", "content": "#### 18"}
```

And then, we use `dataloader.py` to load the gold dataset with the format:

* `question`: problem
* `gold_answer`: ground truth answer
* `cot_solution` (optional): reference CoT

## 2. Primary Metric: Final Answer Accuracy
### Reference:
**[https://arxiv.org/abs/2502.03387](https://arxiv.org/abs/2502.03387)**

### Definition and formula:

```math
Accuracy = \frac{1}{N} \sum_{i=1}^{N} \text{pred}_i == \text{gold}_i
```

### Symbolic equivalence check

We use this github link:
**[https://github.com/huggingface/Math-Verify](https://github.com/huggingface/Math-Verify)** to compute correctness of the final result using:

```python
verify(parse(gold), parse(pred))
```

This automatically checks **mathematical equivalence** in both latex and mathematical expression, rather than raw string match:

```txt
e.g. gold = \frac{1}{2}, pred = 0.5
```

If automatically parsing and verifying fails, the script will directly use normalized exact string match to check the correctness. We use this as the primary metric as this shows the performance of model on each test benchmarks.


## 3. Secondary Metric: Average Word Count

We compute the total number of tokens across `thinking`:

```math
\text{AvgLength} = \frac{1}{N} \sum_{i=1}^{N} \ell_i
```

This metric measures reasoning efficiency of the model outputs. Lower average word count of model output means higher reasoning efficiency as model uses less tokens to get the correct answer.

## 4. Third Metric (Optional): LLM Critic Score
### Reference:
**[https://arxiv.org/abs/2502.03387](https://arxiv.org/abs/2502.03387)**

### Methods
If `--enable-critic` is used, we evaluate reasoning quality by using **[gpt5-nano](https://platform.openai.com/docs/models/gpt-5-nano)** to critic the model reasoning quality by giving scores with range 0 to 5. For each example with gold CoT, we pass **problem**, **gold CoT**, and **model CoT** (`thinking`) to a critic model (using gpt5-nano) and then get the result:
```json
{"score": x}
```

where x is in range **[1, 5]**:

* **5:** excellent reasoning, logically aligned
* **3:** partially correct, some issues
* **1:** incorrect or incoherent reasoning

### Definition and formula:

```math
\text{LLMScore} = \frac{1}{|S|} \sum_{i \in S} s_i
```

Note: We only use top 50 samples for llm critic.

## 5. Examples

### 5.1 Basic usage (Accuracy + Length)

```bash
python score.py \
  --dataset gsm8k \
  --pred-path outputs/gsm8k_predictions.jsonl \
  --data-root data
```

**Example output:**

```
Loaded 1319 gold samples!
Accuracy: 0.7324  (967/1319)
Average word count: 205.37
LLM critic disabled.
```

---

### 5.2 Enabling LLM critic evaluation

First set your API key:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Then run:

```bash
python score.py \
  --dataset gsm8k \
  --pred-path outputs/gsm8k_predictions.jsonl \
  --data-root data \
  --enable-critic \
  --critic-max-samples 50
```

**Example output:**

```
Loaded 1319 gold samples!
Accuracy: 0.7324
Average word count: 205.37
Evaluating 1/50
...
Evaluating 50/50
LLM Critic avg score: 4.18
```

---

### 5.3 Other dataset

For other dataset, just change the arguments with the specific dataset you want to test.

---

## 6. Result for Baseline & Strong-Baseline

   Dataset  | Metric        | Baseline |    Strong-Baseline   |    Extension-1    |
| --------- | ------------- | -------- | -------------------- | ----------------- |
| gsm8k     | **Accuracy**  | 0.5512   | 0.5390               | 0.5739            |
| gsm8k     | **Word Count**| 282.73   | 319.75               | 216.42            |
| gsm8k     | **Critic**    | 2.333    | 3.11                 | 3.89              |
| math500   | **Accuracy**  | 0.1880   | 0.1980               | 0.2000            |
| math500   | **Word Count**| 234.29   | 198.96               | 157.12            |
| math500   | **Critic**    | 2.280    | 2.360                | 2.550             |
| aime      | **Accuracy**  | 0        | 0                    | 0                 |
| aime      | **Word Count**| 257.23   | 453.63               | 245.47            |
| aime      | **Critic**    | 1.067    | 1.082                | 1.071             |
| amc23     | **Accuracy**  | 0.125    | 0.1750               | 0.1500            |
| amc23     | **Word Count**| 274.23   | 376.44               | 245.45            |
| amc23     | **Critic**    | -        | -                    |                   |