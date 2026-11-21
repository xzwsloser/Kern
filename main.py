from model.model_kern import KernConfig, KernCausalLM

if __name__ == '__main__':
    config = KernConfig()
    model = KernCausalLM(config)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"ffn_hidden_size={config.ffn_hidden_size}, vocab_size={config.vocab_size}")
    print(f"total parameters: {params:,} ({params / 1e6:.3f}M)")
    print(f"trainable parameters: {trainable:,} ({trainable / 1e6:.3f}M)")
