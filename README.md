# 1.58 Bit Llama Model
Initial implementation of 1.58-bit Llama Model following the reference paper: https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

In this paper, they outline the code changes necessary to make 1.58 bit ternary quantized training work. In this repo + my transformers fork, I have implemented the changes necessary to make this work. The main changes are in the `transformers` repo, where I have added the `BitLlamaModel` and `BitLlamaForCausalLM` classes. 

![Training code changes](code_from_paper.png)

![Training hyperparams](training_config.png)

See my [transformers repo](https://github.com/bjoernpl/transformers/tree/add_bitllama/src/transformers/models/bitllama), the [uploaded initialized model](https://huggingface.co/bjoernp/micro-bitllama) or `configuration_bitllama.py` and `modeling_bitllama.py` for the modeling implementation.

I've also included a basic axolotl toy pretraining config (`pretraining_bitllama.yaml`) based on a small model that I initialized and [uploaded to HF](https://huggingface.co/bjoernp/micro-bitllama) with the modelling code. You can use this to test the training. Currently, the training is not working as expected as I'm getting nan grads:
![alt text](nan_grads.png)

## Installation
```bash
git clone git@github.com:bjoernpl/transformers.git@add_bitllama
cd transformers
pip install -e .
```

For optional training with axolotl, you can install the following:
```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging
pip3 install -e '.[flash-attn]'
```


## Notes
This is absolutely WIP and experimental. Contributions and ideas towards fixing the nan grads are welcome. This also does not include the custom kernels mentioned (but not described) in the paper.
