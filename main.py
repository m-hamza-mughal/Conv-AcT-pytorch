#!/usr/bin/env python
import os
import argparse
import torch
import shutil
import logging
import sys

from conv_act.data import load_dataset
from conv_act.model import ConvAcTransformer
from conv_act.train import train_model
from conv_act.utils import read_yaml

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['TORCH_HOME'] = "model_cache/"

def main(config):
    # data pipeline
    logging.info("Loading Data")
    train_loader, val_loader, test_loader = load_dataset(
        video_path=config["DATA_VIDEO_PATH"],
        label_dir=config["DATA_ANNOTATION_PATH"],
        num_samples=config["DATA_NUM_SAMPLES"],
        video_dim=config["DATA_RES"], 
        chunk_length=config["VIDEO_CHUNK_LENGTH"], 
        num_frames=config["NUM_FRAMES"],
        batch_size=config["BATCH_SIZE"],
        num_workers=config['DATA_NUM_WORKERS']
    )
    
    # modelling
    logging.info("Loading Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAcTransformer(
        d_model=config['FEATURE_DIM'], 
        attention_heads=config['NUM_ATT_HEADS'], 
        num_layers=config['NUM_TRANSFORMER_LAYERS'], 
        num_classes=config['NUM_CLASSES'], 
        num_frames=config['NUM_FRAMES'],
        feature_extractor_name=config["FEAT_EXTRACTOR"]
    )
    # print(model)
    model = model.to(device)

    logging.info("Training")
    _, stats = train_model(
        model,
        train_loader, 
        val_loader, 
        model_name=config["MODEL_PATH"], 
        num_epochs=config["N_EPOCHS"],
        learning_rate=config["LR"],
        finetune=config["FINETUNE"]
    )

    logging.info(str(stats))



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train and Evaluation Pipeline')
    parser.add_argument('--config', default='config/ucf50_128.yaml', type=str, help='Config path', required=False)
    args = parser.parse_args()
    config = read_yaml(args.config)

    logging.info("Using device: cuda" if torch.cuda.is_available() else "Using device: cpu")

    dir_path = os.path.join("./logs/", config["MODEL_PATH"])
    if not os.path.exists(dir_path):
        logging.info("Making dir")
        os.mkdir(dir_path)
    else:
        logging.info(f"{dir_path} already exists. Overwriting files in it..")
    
    shutil.copyfile(args.config, os.path.join(dir_path, "config.yaml"))

    main(config)
