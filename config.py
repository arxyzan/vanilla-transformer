config = {
    'name': 'de_en_translation',
    'embed_dim': 256,
    'n_blocks': 3,
    'n_heads': 8,
    'ff_hid_dim': 512,
    'dropout': 0.1,
    'max_length': 100,
    'device': 'cuda',
    'lr': 0.0005,
    'clip': 1,
    'log_dir': 'logs',
    'weights_dir': 'weights',
    'save_interval': 1,
    'train_batch_size': 128,
    'val_batch_size': 128,
    'epochs': 10
}
