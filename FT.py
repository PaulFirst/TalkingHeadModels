import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib

import numpy as np

from dataset.dataset_class import FineTuningImagesDataset, FineTuningVideoDataset
from network.model import *
from loss.loss_discriminator import *
from loss.loss_generator import *

from params.params import K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape


class FT:
    def __init__(self, path_to_images='', path_to_video='', path_to_embedding='', epochs=10):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path_to_embedding = path_to_embedding
        self.path_to_save = 'finetuned_model.tar'
        self.path_to_video = path_to_video
        self.path_to_images = path_to_images
        self.dataset = None
        self.dataLoader = None
        self.path_to_chkpt = path_to_chkpt
        self.epochs = epochs
        self.G = None
        self.D = None
        self.e_hat = None

    def createImageDataset(self):
        self.dataset = FineTuningImagesDataset(self.path_to_images, self.device)
        self.dataLoader = DataLoader(self.dataset, batch_size=2, shuffle=False)

    def createVideoDataset(self):
        self.dataset = FineTuningVideoDataset(self.path_to_video, self.device)
        self.dataLoader = DataLoader(self.dataset, batch_size=2, shuffle=False)

    def loadModel(self):
        self.e_hat = torch.load(self.path_to_embedding, map_location='cpu')
        self.e_hat = self.e_hat['e_hat']

        self.G = Generator(256, finetuning=True, e_finetuning=self.e_hat)
        self.D = Discriminator(self.dataset.__len__(), path_to_Wi, finetuning=True, e_finetuning=self.e_hat)
        self.G.train()
        self.D.train()

        if not os.path.isfile(self.path_to_chkpt):
            print('ERROR: cannot find checkpoint')
        if os.path.isfile(self.path_to_save):
            self.path_to_chkpt = self.path_to_save

        checkpoint = torch.load(self.path_to_chkpt, map_location='cpu')
        checkpoint['D_state_dict']['W_i'] = torch.rand(512, 2)  # change W_i for finetuning

        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'], strict=False)

    def fineTuning(self):
        optimizerG = optim.Adam(params=self.G.parameters(), lr=5e-5)
        optimizerD = optim.Adam(params=self.D.parameters(), lr=2e-4)

        criterionG = LossGF(VGGFace_body_path='Pytorch_VGGFACE_IR.py',
                            VGGFace_weight_path='Pytorch_VGGFACE.pth', device=self.device)
        criterionDreal = LossDSCreal()
        criterionDfake = LossDSCfake()

        epochCurrent = epoch = i_batch = 0
        lossesG = []
        lossesD = []
        i_batch_current = 0
        for epoch in range(self.epochs):
            for i_batch, (x, g_y) in enumerate(self.dataLoader):
                with torch.autograd.enable_grad():

                    optimizerG.zero_grad()
                    optimizerD.zero_grad()

                    x_hat = self.G(g_y, self.e_hat)
                    r_hat, D_hat_res_list = self.D(x_hat, g_y, i=0)
                    with torch.no_grad():
                        r, D_res_list = self.D(x, g_y, i=0)

                    lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list)

                    lossG.backward(retain_graph=False)
                    optimizerG.step()

                    # train D
                    optimizerD.zero_grad()
                    x_hat.detach_().requires_grad_()
                    r_hat, D_hat_res_list = self.D(x_hat, g_y, i=0)
                    r, D_res_list = self.D(x, g_y, i=0)

                    lossDfake = criterionDfake(r_hat)
                    lossDreal = criterionDreal(r)

                    lossD = lossDreal + lossDfake
                    lossD.backward(retain_graph=False)
                    optimizerD.step()

                    # train D again
                    optimizerG.zero_grad()
                    optimizerD.zero_grad()
                    r_hat, D_hat_res_list = self.D(x_hat, g_y, i=0)
                    r, D_res_list = self.D(x, g_y, i=0)

                    lossDfake = criterionDfake(r_hat)
                    lossDreal = criterionDreal(r)

                    lossD = lossDreal + lossDfake
                    lossD.backward(retain_graph=False)
                    optimizerD.step()


        plt.clf()
        out = (x_hat[0] * 255).transpose(0, 2)
        for img_no in range(1, x_hat.shape[0]):
            out = torch.cat((out, (x_hat[img_no] * 255).transpose(0, 2)), dim=1)
        out = out.type(torch.int32).to('cpu').numpy()
        fig = out

        plt.clf()
        out = (x[0] * 255).transpose(0, 2)
        for img_no in range(1, x.shape[0]):
            out = torch.cat((out, (x[img_no] * 255).transpose(0, 2)), dim=1)
        out = out.type(torch.int32).to('cpu').numpy()
        fig = np.concatenate((fig, out), 0)

        plt.clf()
        out = (g_y[0] * 255).transpose(0, 2)
        for img_no in range(1, g_y.shape[0]):
            out = torch.cat((out, (g_y[img_no] * 255).transpose(0, 2)), dim=1)
        out = out.type(torch.int32).to('cpu').numpy()

        fig = np.concatenate((fig, out), 0)
        plt.imshow(fig)
        plt.xticks([])
        plt.yticks([])
        plt.draw()
        plt.pause(0.001)
        plt.savefig('fig.png')

        torch.save({
            'epoch': epoch,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
        }, self.path_to_save)

