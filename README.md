# Causal Tracing of "Answer-First" Bias in Math Reasoning

## 📌 Project Overview
This project investigates the **faithfulness** of Chain-of-Thought (CoT) reasoning in Large Language Models. Specifically, we test the **"Answer-First" Hypothesis**:
> *Does the model "know" or "decide" the final answer in its hidden states **before** it generates the reasoning steps?*

If the answer (e.g., "42") is present in the residual stream at the end of the question (before reasoning starts), it suggests the reasoning chain is **post-hoc justification** (unfaithful). If the answer is absent and only emerges *during* the reasoning process, the CoT is likely **necessary** for the computation (faithful).

## 🔬 Methodology: Logit Lens
We use the **Logit Lens** technique to "x-ray" the model's internal activations without training any probes.

1.  **Input**: A math problem from GSM8K (e.g., "If John has 2 apples...").
2.  **Probe Point**: The **final token of the question input** (Layer 0 to Layer $L$). This is the exact moment before the model generates the first token of its response.
3.  **Decoding**: At each layer $i$, we take the hidden state $h_i$ and project it directly to the vocabulary using the model's own unembedding matrix $W_U$:
    $$P(token) = \text{softmax}(LayerNorm(h_i) \cdot W_U^T)$$
4.  **Metric**: We track the probability assigned specifically to the **ground truth answer** token.

## 🧪 Experiment Phase 1: Pre-Reasoning Check
We ran a controlled experiment to see if the answer exists *before* reasoning.

*   **Model**: `Qwen/Qwen2.5-Math-1.5B-Instruct`
*   **Dataset**: GSM8K (Test Split)
*   **Sample Size**: 100 samples (filtered for single-token numerical answers).
*   **Compute**: Local Inference (MPS/CUDA).

### 📊 Results & Interpretation
**Key Finding: 0% Early Emergence (High Faithfulness)**

Across **100 samples**, the model showed **negligible probability (~0.00%)** of predicting the correct answer at the end of the question.

*   **Why is 0% probability expected/good?**
    At the end of the question, the model is predicting the *next token*. For a faithful CoT model, the next token should be the start of reasoning (e.g., "Let's", "To", "First"), NOT the answer (e.g., "42").
    *   **Faithful Model**: Predicts "Let's" (Prob near 1.0), Predicts "42" (Prob near 0.0).
    *   **Unfaithful Model**: Might covertly predict "42" in a middle layer (Prob > 0.0) but then outputs "Let's" in the final layer.

*   **Layer-wise Analysis**: As seen in the plot below, the probability of the correct answer remains flat and near zero across all 28 layers. This confirms the model does not "secretly know" the answer. It genuinely requires the reasoning chain to traverse the solution space.

![Average Probability](avg_probs.png)
*Figure 1: Average probability of the correct answer token across layers (at the last token of the prompt).*

![Heatmap](heatmap.png)
*Figure 2: Heatmap of answer probability for all 100 samples. The darkness confirms that no individual sample exhibited "Answer-First" bias.*

## 🚀 Usage

### 1. Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/ranabir/Faithful-Lens.git
cd Faithful-Lens
pip install -r requirements.txt
```

### 2. Configuration
Edit `config.yaml` to adjust parameters (e.g., switch model, increase sample size):
```yaml
model:
  name: "Qwen/Qwen2.5-Math-1.5B-Instruct"
dataset:
  num_samples: 100
inspection:
  layers_to_inspect: [15] # Inspect specific layers
```

### 3. Run Experiment
Execute the analysis script:
```bash
python -m src.run_experiment
```
This will:
1.  Load the model and dataset.
2.  Run the Logit Lens analysis.
3.  Save results to `results/analysis_results.json`.

### 4. Visualize Results
Generate plots (saved to root directory):
```bash
python src/visualize.py
```

### 5. Inspect Specific Layers
To see what the model *is* predicting (since it's not the answer), run:
```bash
python -m src.inspect_layer
```

## 🔜 Phase 2: Middle-of-Reasoning Trace
The next phase of this project involves **dynamic tracing**:
*   Hooking into the generation loop.
*   Applying Logit Lens at **every token step** of the generated Chain-of-Thought.
*   **Goal**: Pinpoint the exact "Aha!" moment where the answer probability spikes (e.g., does it happen 5 tokens before the final number is generated?).

## 📁 Repository Structure
*   `src/`: Core implementation.
    *   `logit_lens.py`: Minimal class for hooking and decoding residual streams.
    *   `analysis.py`: Experiment logic.
    *   `inspect_layer.py`: Tool to inspect top-k predictions at specific layers.
    *   `data_loader.py`: GSM8K processing and answer extraction.
*   `notebooks/`: Prototype notebooks.
*   `results/`: JSON data storage.
*   `config.yaml`: Central configuration.
