import torch
import cv2
from matplotlib import pyplot as plt

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from webcam_demo.webcam_extraction_conversion import *
from dataset.video_extraction_conversion import generate_landmarks

from params.params import path_to_chkpt
from tqdm import tqdm

class Inference:
    def __init__(self, path_to_embedding, path_to_image='', path_to_video=''):
        super().__init__()
        self.path_to_model_weights = 'finetuned_model.tar'
        self.path_to_embedding = path_to_embedding
        self.path_to_video = path_to_video
        self.path_to_image = path_to_image
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint = torch.load('finetuned_model.tar', map_location='cpu')
        self.e_hat = torch.load(self.path_to_embedding, map_location='cpu')
        self.e_hat = self.e_hat['e_hat'].to(self.device)
        self.G = Generator(256, finetuning=True, e_finetuning=self.e_hat)
        self.G.eval()
        self.G.load_state_dict(self.checkpoint['G_state_dict'])
        self.G.to(self.device)

    def generateImage(self):
        with torch.no_grad():
            x, g_y = generate_landmarks(image_path=self.path_to_image,device=self.device, pad=50)
            g_y = g_y.unsqueeze(0) / 255
            x = x.unsqueeze(0) / 255
            x_hat = self.G(g_y, self.e_hat)
            plt.clf()
            out1 = x_hat.transpose(1, 3)[0]
            out1 = out1.to('cpu').numpy()
            out2 = x.transpose(1, 3)[0]
            out2 = out2.to('cpu').numpy()
            out3 = g_y.transpose(1, 3)[0]
            out3 = out3.to('cpu').numpy()

            fake = out1 * 255
            me = cv2.cvtColor(out2 * 255, cv2.COLOR_BGR2RGB)
            landmark = cv2.cvtColor(out3 * 255, cv2.COLOR_BGR2RGB)
            img = np.concatenate((me, landmark, fake), axis=1)
            img = img.astype('uint8')
            fake = fake.astype('uint8')
            #video.write(img)
            plt.imshow(fake)
            plt.draw()
            plt.savefig('result_image.png')

    def generateVideo(self):
        cap = cv2.VideoCapture(self.path_to_video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ret = True
        i = 0
        size = (256 * 3, 256)
        video = cv2.VideoWriter('result_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        with torch.no_grad():
            while ret:
                x, g_y, ret = generate_landmarks(cap=cap, device=self.device, pad=50)
                if ret:
                    g_y = g_y.unsqueeze(0) / 255
                    x = x.unsqueeze(0) / 255

                    x_hat = self.G(g_y, self.e_hat)

                    plt.clf()
                    out1 = x_hat.transpose(1, 3)[0]
                    out1 = out1.to('cpu').numpy()
                    out2 = x.transpose(1, 3)[0]
                    out2 = out2.to('cpu').numpy()
                    out3 = g_y.transpose(1, 3)[0]
                    out3 = out3.to('cpu').numpy()


                    fake = out1 * 255
                    me = cv2.cvtColor(out2 * 255, cv2.COLOR_BGR2RGB)
                    landmark = cv2.cvtColor(out3 * 255, cv2.COLOR_BGR2RGB)
                    img = np.concatenate((me, landmark, fake), axis=1)
                    img = img.astype('uint8')
                    fake = fake.astype('uint8')
                    video.write(fake)

                    i += 1
                    print(i, '/', n_frames)
        cap.release()
        video.release()
