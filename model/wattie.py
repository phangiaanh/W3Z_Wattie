import torch
import torch.nn as nn
import math
import numpy as np
import os
from huggingface_hub import hf_hub_download
from model.extractor import VitExtractor
from local_utils.geometry import rot6d_to_rotmat
import argparse

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

class Wattie(nn.Module):
    def __init__(self, opts):
        super(Wattie, self).__init__()
        self.opts = opts
        if self.opts.animalType == "equidae":
            self.nshape = 9
            self.npose = 35
        elif self.opts.animalType == "canidae":
            self.nshape = 9
            self.npose = 34
        else:
            self.nshape = 20
            self.npose = 32

        self.ncam = 3
        self.nbbox = 3
        self.imgsize = opts.imgsize
        img_feat_num = 384
        kernel_size = 7 if self.imgsize == 224 else 8

        self.vit_feat_dim = img_feat_num
        text_embedding = 640
        fc1_feat_num = 1024
        fc2_feat_num = text_embedding
        final_feat_num = fc2_feat_num

        self.key_encoder_shape = nn.Sequential(
                nn.Conv2d(img_feat_num, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2), # Output size: [2, 256, 16, 16]
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2), # Output size: [2, 128, 8, 8]
                nn.Conv2d(128, text_embedding, kernel_size=kernel_size)  #  # Output size: [2, 640, 1, 1]
            )
        self.key_encoder_pose = nn.Sequential(
            nn.Conv2d(img_feat_num, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [2, 256, 16, 16]
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [2, 128, 8, 8]
            nn.Conv2d(128, text_embedding, kernel_size=kernel_size)  # # Output size: [2, 640, 1, 1]
        )
        self.key_encoder_cam = nn.Sequential(
            nn.Conv2d(img_feat_num, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [2, 256, 16, 16]
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [2, 128, 8, 8]
            nn.Conv2d(128, text_embedding, kernel_size=kernel_size)  # # Output size: [2, 640, 1, 1]
        )

        self.drop1 = nn.Dropout()
        self.fc_decpose1 = nn.Linear(final_feat_num + self.npose * 6, fc1_feat_num)
        self.fc_decpose2 = nn.Linear(fc1_feat_num, final_feat_num)
        self.decpose = nn.Linear(final_feat_num, self.npose * 6)

        self.fc_decshape1 = nn.Linear(final_feat_num + self.nshape, fc1_feat_num)
        self.fc_decshape2 = nn.Linear( fc1_feat_num, final_feat_num)
        self.decshape = nn.Linear(final_feat_num, self.nshape)

        self.fc_deccam1 = nn.Linear(final_feat_num + self.ncam + 6, fc1_feat_num)
        self.fc_deccam2 = nn.Linear(fc1_feat_num, final_feat_num)
        self.deccam = nn.Linear(final_feat_num, self.ncam)
        self.decroot = nn.Linear(final_feat_num, 1 * 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

        mean_pose_path = hf_hub_download(
            repo_id="WatermelonHCMUT/SamplePose",
            filename="6D_meanpose.npz",
            local_dir="./temp"
        )
        
        self.encoder = VitExtractor(model_name="dino_vits8", frozen = False)

        mean_params = np.load(mean_pose_path, allow_pickle=True)['mean_pose'].astype(
            np.float32)[:self.npose + 1, :]
        mean_params[0, :] = [-0.0517, 0.9979, -0.2873, -0.0517, -0.9564, -0.0384]
        init_pose = torch.reshape(torch.from_numpy(mean_params.copy()).unsqueeze(0), [1, -1])
        init_shape = torch.from_numpy(np.zeros(self.nshape)).unsqueeze(0).type(torch.float32)
        init_cam = torch.from_numpy(np.array([0.6,0.0,0.0])).unsqueeze(0).type(torch.float32)
        
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def obtain_latent_space(self, x):
        
        ps = self.encoder.get_patch_size()
        pw, ph = self.imgsize // ps, self.imgsize // ps

        b = x.shape[0]
        xf = self.encoder.get_keys_from_input(x, layer_num=11)[:,:,1:,:].permute(0, 1, 3, 2).reshape(b, self.vit_feat_dim, ph, pw)
        xf_shape = self.key_encoder_shape(xf).view(b, -1)
        xf_pose = self.key_encoder_pose(xf).view(b, -1)
        xf_cam = self.key_encoder_cam(xf).view(b, -1)
        return xf_shape, xf_pose, xf_cam
    
    def predict_shape(self, x, init_shape):
        pred_shape = init_shape
        for i in range(3):
            xc = torch.cat([x, pred_shape], 1)
            xc = self.fc_decshape1(xc)
            xc = self.drop1(xc)
            xc = self.fc_decshape2(xc)
            xc = self.drop1(xc)
            pred_shape = self.decshape(xc) + pred_shape
        return pred_shape

    def predict_pose(self, x, init_pose):
        pred_pose = init_pose
        for i in range(3):
            xc = torch.cat([x, pred_pose], 1)
            xc = self.fc_decpose1(xc)
            xc = self.drop1(xc)
            xc = self.fc_decpose2(xc)
            xc = self.drop1(xc)
            pred_pose = self.decpose(xc) + pred_pose
        return pred_pose  # , pred_rotmat

    def predict_cam(self, x, init_cam, init_root):
        pred_cam = init_cam
        pred_root = init_root
        for i in range(3):
            xc = torch.cat([x, pred_cam, pred_root], 1)
            xc = self.fc_deccam1(xc)
            xc = self.drop1(xc)
            xc = self.fc_deccam2(xc)
            xc = self.drop1(xc)
            pred_cam = self.deccam(xc) + pred_cam
            pred_root = self.decroot(xc) + pred_root
        return pred_cam, pred_root
    
    def predict_from_latent(self, xf_shape, xf_pose, xf_cam):
        # predict
        batch_size = xf_shape.shape[0]
        pred_shape = self.predict_shape(xf_shape, init_shape=self.init_shape.expand(batch_size, -1))
        pred_pose = self.predict_pose(xf_pose, init_pose=self.init_pose[0, 6:].expand(batch_size, -1))
        pred_cam, pred_root = self.predict_cam(xf_cam, init_cam=self.init_cam.expand(batch_size, -1),
                                               init_root=self.init_pose[0, :6].expand(batch_size, -1))
        pred_6D = torch.cat([pred_root, pred_pose], 1)
        pred_rotmat = rot6d_to_rotmat(pred_6D).view(batch_size, self.npose + 1, 3, 3)   
        pred_cam = torch.cat([torch.tensor([[0.6]]).repeat(batch_size, 1).float().to(pred_pose.device), pred_cam],dim=1)
        return pred_rotmat, pred_shape, pred_cam

    def forward(self, x):
        print(f"Before1")
        xf_shape, xf_pose, xf_cam = self.obtain_latent_space(x)
        print(f"Before2")
        pred_rotmat, pred_shape, pred_cam = self.predict_from_latent(xf_shape, xf_pose, xf_cam)
        print(f"Before3")
        return pred_rotmat, pred_shape, pred_cam, xf_shape, xf_pose, xf_cam
        
        
if __name__ == "__main__":
    opts = parse_args()
    model = Wattie(opts)
    inp = torch.rand(128, 3, 256, 256)
    out = model(inp, )
    print('rot', out[0].shape, 'shape', out[1].shape, 'cam', out[2].shape)