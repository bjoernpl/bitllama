from transformers.models.bitllama import BitLlamaForCausalLM, LlamaConfig

model_config = LlamaConfig(
    # Config for a tiny model model with 1.62M parameters
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=512,
    initializer_range=0.02,
    intermediate_size=1365,
    max_position_embeddings=32000,
    num_attention_heads=8,
    num_hidden_layers=12,
    num_key_value_heads=4,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=32000,
)

model = BitLlamaForCausalLM._from_config(model_config)
model.push_to_hub("bjoernp/micro-bitllama")