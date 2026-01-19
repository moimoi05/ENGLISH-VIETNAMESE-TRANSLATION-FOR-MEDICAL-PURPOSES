config = {
    "train_path": "../Data_Medi/med_train",
    "val_path": "../Data_Medi/med_val",

    "seq_len": 350,
    "d_model": 512,

    "batch_size": 128,
    "num_epochs": 15,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "label_smoothing": 0.05,

    "pretrained_ckpt": "/content/drive/MyDrive/DL/Transformer from Scratch/Finetune/weights_finetune_300k/finetune_best.pt",
    "save_dir": "weights_finetune_300k",

    "print_every": 100
}
