# AVSEC-3-Challenge
Audio-Visual Speech Enhancement Challenge (AVSE) 2024


## Requirements
* Python >= 3.8
* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
* [Decord](https://github.com/dmlc/decord)

```bash
# You can install all requirements using pip install -r requirements.txt
```

## Usage
Update DATA_ROOT in config.py 
```bash
# Expected folder structure
avsec3_data_root
├── dev
│   ├── lips
│   │   └── S37890_silent.mp4
│   └── scenes
│       ├── S37890_interferer.wav
│       ├── S37890_mixed.wav
│       ├── S37890_silent.mp4
│       └── S37890_target.wav
├── train
│   ├── lips
│   │   └── S34526_silent.mp4
│   └── scenes
│       ├── S34526_interferer.wav
│       ├── S34526_mixed.wav
│       ├── S34526_silent.mp4
│       └── S34526_target.wav
└── eval 
    ├── lips
    │   └── S34526_silent.mp4
    └── scenes
        ├── S34526_mixed.wav
        └── S34526_silent.mp4
```

### Train
```bash
python train.py --log_dir ./logs --batch_size 2 --lr 0.001 --gpu 1 --max_epochs 20

python train.py --log_dir logs --batch_size 2 --lr 0.001 --gpu 1 --max_epochs_no 50

optional arguments:
  -h, --help            show this help message and exit
  --batch_size 4        Batch size for training
  --lr 0.001               Learning rate for training
  --log_dir LOG_DIR     Path to save tensorboard logs
```

### Test
```bash
usage: test.py [-h] --ckpt_path ./model.pth --save_root ./enhanced --model_uid avse [--dev_set False] [--eval_set True] [--cpu True]

# Model evaluation - dev set
python test.py --ckpt_path pre_model\avse_baseline.ckpt --save_root enhanced --model_uid avse --dev_set True --eval_set False --cpu False

# Model evaluation - val set
python test.py --ckpt_path pre_model\avse_baseline.ckpt --save_root enhanced --model_uid avse --dev_set False --eval_set True --cpu False


optional arguments:
  -h, --help             show this help message and exit
  --ckpt_path CKPT_PATH  Path to model checkpoint
  --save_root SAVE_ROOT  Path to save enhanced audio
  --model_uid MODEL_UID  Folder name to save enhanced audio
  --dev_set True         Evaluate model on dev set
  --eval_set False       Evaluate model on eval set
  --cpu True              Evaluate on CPU (default is GPU)
```

### Evaluate
```bash  
python objective_evaluation.py
```

### Contact Us
For further queries, contact us at:

[Arnav Jain](https://github.com/arnavjain2710) - arnavjain2710@gmail.com 

[Jasmer Singh Sanjotra](https://github.com/TheAlphaJas) - jasmer.sanjotra@gmail.com 

[Harshvardhan Chaudhary](https://github.com/Harshvardhan-To1)
