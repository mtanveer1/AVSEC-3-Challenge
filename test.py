from argparse import ArgumentParser
from os.path import isfile
from os import makedirs
from os.path import join

import soundfile as sf
import torch
from tqdm import tqdm


from config import sampling_rate

from dataset import AVSEDataModule
from model import AVSEModule
from utils import str2bool


def main(args):
    enhanced_root = join(args.save_root, args.model_uid)
    makedirs(args.save_root, exist_ok=True)
    makedirs(enhanced_root, exist_ok=True)
    datamodule = AVSEDataModule(batch_size=1)
    if args.dev_set and args.eval_set:
        raise RuntimeError("Select either dev set or test set")
    elif args.dev_set:
        dataset = datamodule.dev_dataset
    elif args.eval_set:
        dataset = datamodule.eval_dataset
    else:
        raise RuntimeError("Select one of dev set and test set")
    try:
        model = AVSEModule.load_from_checkpoint(args.ckpt_path)
        print("Model loaded")
    except Exception as e:
        raise FileNotFoundError("Cannot load model weights: {}".format(args.ckpt_path))
    if not args.cpu:
        model.to("cuda:0")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            filename = f"{data['scene']}.wav"
            enhanced_path = join(enhanced_root, filename[1:])
            if not isfile(enhanced_path):
                estimated_audio = model.enhance(data).reshape(-1)
                sf.write(enhanced_path, estimated_audio, samplerate=sampling_rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--save_root", type=str, required=True, help="Path to save enhanced audio")
    parser.add_argument("--model_uid", type=str, required=True, help="Folder name to save enhanced audio")
    parser.add_argument("--dev_set", type=str2bool, default=False, help="Evaluate model on dev set")
    parser.add_argument("--eval_set", type=str2bool, default=False, help="Evaluate model on eval set")
    parser.add_argument("--cpu", type=str2bool, required=False, help="Evaluate model on CPU")
    args = parser.parse_args()
    main(args)
