import argparse
import torch
import random
import numpy as np
from W3Z_AnimalPIPE.main import AnimalPipe
from trainer.vanillaupdatetrainer import VanillaUpdateTrainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/wattie/")
    parser.add_argument("--output_path", type=str, default="data/wattie/")
    parser.add_argument('--useSynData', action="store_true", help="True: use syndataset")
    parser.add_argument('--useinterval', type=int, default=8, help='number of interval of the data')
    parser.add_argument("--getPairs", action="store_true", default=True,help="get image pair with label")
    parser.add_argument("--animalType", type=str, default='equidae', help="animal type")

    parser.add_argument('--imgsize', type=int, default=256, help='number of workers')
    parser.add_argument('--background', type=bool, default=False, help='background')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--data_batch_size', type=int, default=2, help='batch size; before is 36')

    parser.add_argument('--save_dir', type=str, default='/home/watermelon/sources/hcmut/W3Z/W3Z_Wattie/save',help='save dir')
    parser.add_argument('--name', type=str, default='test', help='experiment name')
    parser.add_argument('--version', type=str, required=False, help='experiment version')

    parser.add_argument('--lr', type=float, default=5e-05, help='optimizer learning rate')
    parser.add_argument('--max-epochs', type=int, default=1, help='max. number of training epochs')

    parser.add_argument('--ckpt_file', type=str, required=False, help='checkpoint for resuming')
    return parser.parse_args()


def get_syn_dataset(args, device, train_length=1000, valid_length=30):
    dataset = AnimalPipe(args=args, device=device, length=train_length, FLAG='TRAIN', ANIMAL=args.animalType)
    validdataset = AnimalPipe(args=args, device=device, length=valid_length, FLAG='VALID', ANIMAL=args.animalType)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=True,pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(validdataset, batch_size=args.batch_size, num_workers=0, shuffle=True,pin_memory=True, drop_last=True)
    return train_loader, val_loader


def main(args):
    print(args)

    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    
    if args.useSynData:
        if args.getPairs:
            train_length = 320
            valid_length = 32
        else:
            raise NotImplementedError
        
        train_loader, val_loader = get_syn_dataset(args, device, train_length, valid_length)
    else:
        raise NotImplementedError
    
    vanilla = VanillaUpdateTrainer(opts=args)
    logger = TensorBoardLogger(args.save_dir, name=args.name, version=args.version)
    logger.log_hyperparams(vars(args))  # save all (hyper)params

    ckpt_callback = ModelCheckpoint(
        filename='best',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    callbacks_list = [ckpt_callback]
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks_list,
        accelerator=accelerator,
        devices=num_gpus,  
        max_epochs=args.max_epochs,
        gradient_clip_val=None, 
        gradient_clip_algorithm=None, 
        log_every_n_steps=int(len(train_loader) / num_gpus),
        enable_progress_bar=True,
        strategy='ddp' if num_gpus > 1 else 'auto',
    )

    # train model
    trainer.fit(
        vanilla,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.ckpt_file
    )
    

if __name__ == "__main__":

    import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)

    args = parse_args()
    main(args)

    




