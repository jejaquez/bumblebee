# 🐝 Bumblebee: Fuzzing the Mind of AI and Machines

**Bumblebee** is an AI fuzzing tool that audibly and visually inspects transformer models to detect behavioral instabilities, layer by layer. Inspired by the sound of fuzzing and the buzz of bees, this tool helps researchers and engineers explore model robustness in a new light.


## 🚀 Features

- Hooks into HuggingFace transformer models
- Generates visual and JSON layer-level inspection reports
- Supports fuzzing across all attention and MLP layers
- Designed for research into interpretability and robustness
- Multiple Fuzzing Modes: Includes `gaussian`, `bitflip`, and `dropout` fuzzing techniques to perturb activations in different ways.
- Customizable Hook Targets: Fuzzing can be applied to specific transformer internals.
- Activation-Level Perturbation: Directly injects noise into internal activation streams, allowing precise robustness testing across transformer layers.
- Differential Similarity Analysis: Measures how much the model output diverges from the original using token-wise diff comparisons and similarity scoring.
- Structured JSON Output: Saves comprehensive results — including prompts, configurations, and analysis — to timestamped `.json` reports.
- Command-Line Interface: Easy-to-use CLI interface for specifying model, input prompt, and fuzzing mode.
- Prompt-Level Output Comparison: Evaluates the effects of perturbations by comparing clean vs. corrupted completions.
- Modular Fuzzing Engine: Easily extensible: add new fuzzing strategies by updating the `fuzz_methods` dictionary.

## 🔧 Usage

```bash
python bumblebee.py --model gpt2
