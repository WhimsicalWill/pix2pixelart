import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import wandb 

from src.modules import Generator, Discriminator, SiameseNet
from tqdm import tqdm

wandb.init('TravelGAN', name='PixelArt')

class TravelGan:
    def __init__(self, config, logger):
        self.logger = logger 
        self.config = config 
        self.device = self.config['device']
        self.dis = Discriminator(config['in_channels'], self.device, num_feat=config['num_feat'], num_repeat=config['num_repeat']).to(self.device)
        self.gen = Generator(config['in_channels'], num_feat=config['num_feat'], num_res=config['num_res']).to(self.device)
        self.siamese = SiameseNet(config['image_size'], config['in_channels'], num_feat=config['num_feat'], 
                                  num_repeat=config['num_repeat'], gamma=config['gamma']).to(self.device)
        wandb.watch([self.gen, self.dis, self.siamese], log='all')
        
        #optimizer
        gen_params = list(self.gen.parameters()) + list(self.siamese.parameters())
        self.opt_gen = optim.Adam(gen_params, lr=config['gen_lr'], betas=(config['gbeta1'], config['gbeta2']))
        self.opt_dis = optim.Adam(self.dis.parameters(), lr=config['dis_lr'], betas=(config['dbeta1'], config['dbeta2']))
        
        # scheduler 
        self.gen_scheduler = optim.lr_scheduler.StepLR(self.opt_gen, config['step_size'], gamma=0.1)
        self.dis_scheduler = optim.lr_scheduler.StepLR(self.opt_dis, config['step_size'], gamma=0.1)

    # call generator multiple times on larger image grid
    # will call generator size_reduce ** 2 times
    def gen_grid(self, x_a):
        x_ab = torch.zeros_like(x_a).to(self.device) # shape (B, 3, H, W)
        dh, dw = x_ab.shape[2] // self.config['size_reduce'], x_ab.shape[3] // self.config['size_reduce']
        for y in range(self.config['size_reduce']):
            for x in range(self.config['size_reduce']):
                x_ab[:, :, dh*y:dh*y+dh, dw*x: dw*x+dw] = self.gen(x_a[:, :, dh*y:dh*y+dh, dw*x: dw*x+dw]) 
        return x_ab

    def _train_epoch(self, loaderA, loaderB, epoch):
        for i, (x_a, x_b) in enumerate(zip(loaderA, loaderB)):
            # zip only creates as many pairs as it can; TODO: make sure overflow is random each time

            global_step = len(loaderB) * epoch + i
            
            if isinstance(x_a, (tuple, list)):
                x_a = x_a[0]
            if isinstance(x_b, (tuple, list)):
                x_b = x_b[0]
            
            x_a = x_a.to(self.device)
            x_b = x_b.to(self.device)
            
            #===============================
            # Dis Update 
            #===============================
            self.opt_dis.zero_grad()
            x_ab = self.gen_grid(x_a)

            dis_loss = self.dis.calc_dis_loss(x_b, x_ab.detach())
            dis_loss.backward()
            self.opt_dis.step()
            
            if global_step  % self.config['iter_log'] == 0:
                self.logger.add_scalar('dis_loss', dis_loss.item(), global_step)

            #===============================
            # Gen Update 
            #===============================
            torch.autograd.set_detect_anomaly(True)
            self.opt_gen.zero_grad()
            gen_adv_loss = self.dis.calc_gen_loss(x_ab)
            gen_siamese_loss = self.siamese.calc_loss(x_a, x_ab)
            color_loss = self.dis.calc_color_loss(x_a.detach().clone(), x_ab) # detach/clone to avoid backprop/modifying

            gen_loss = self.config['gen_adv_loss_w'] * gen_adv_loss + \
                        self.config['siamese_loss_w'] * gen_siamese_loss + \
                        self.config['color_loss_w'] * color_loss
            
            gen_loss.backward()
            self.opt_gen.step()
            
            if global_step % self.config['iter_log'] == 0 :
                self.logger.add_scalar('gen_loss', gen_loss.item(), global_step)
                self.logger.add_scalar('gen_adv_loss', gen_adv_loss.item(), global_step)
                self.logger.add_scalar('color_loss', color_loss.item(), global_step)
                self.logger.add_scalar('siamese_loss', gen_siamese_loss.item(), global_step)
            
            if global_step % self.config['iter_sample'] == 0:
                self.sample(x_a, x_b, x_ab, global_step)
            
    def train(self, loaderA, loaderB):
        for i in tqdm(range(self.config['epochs'])):
            self.gen.train()
            self._train_epoch(loaderA, loaderB, i)
            self.gen_scheduler.step()
            self.dis_scheduler.step()
            
            if i % self.config['checkpoint_iter'] == 0:
                self.save(i)

    def sample(self, x_a, x_b, x_ab, step):
        self.gen.eval()
        x_ab = self.gen_grid(x_a)
        self.logger.add_image('real images', x_a, step)
        self.logger.add_image('pix images', x_b, step)
        self.logger.add_image('real->pix images', x_ab, step)
        self.gen.train() # re-enable training mode

    def save(self, iter): 
        torch.save({'gen' : self.gen.state_dict(),
                    'dis' : self.dis.state_dict(),
                    'siamese' : self.siamese.state_dict(),
                    'gen_opt' : self.opt_gen.state_dict(),
                    'dis_opt' : self.opt_dis.state_dict(),
                    'gen_scheduler' : self.gen_scheduler.state_dict(),
                    'dis_scheduler' : self.dis_scheduler.state_dict()
        }, f'logs/checkpoints/checkpoint_n{iter}.pt')
        
    def load(self):
        checkpoint = torch.load(self.config['checkpoint_path'])
        self.gen.load_state_dict(checkpoint['gen'])
        self.dis.load_state_dict(checkpoint['dis'])
        self.siamese.load_state_dict(checkpoint['siamese'])
        self.opt_dis.load_state_dict(checkpoint['dis_opt'])
        self.opt_gen.load_state_dict(checkpoint['gen_opt'])
        self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
        self.dis_scheduler.load_state_dict(checkpoint['dis_scheduler'])