from parser import CustomParser
from data import get_dataloader
from model import CycleGAN
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import torch

opt = CustomParser().get_parser()
opt.isTrain = True

dataloader = get_dataloader(opt)
model = CycleGAN(opt)
model.setup()

summary = SummaryWriter()
total_iter = 0

# training
for epoch in tqdm(range(1, 201), desc="Process"):
    epoch_start_time = time.time()

    for i, batch in enumerate(dataloader):
        start_time = time.time()
        total_iter += 1 # batch_size = 1
        
        # update weights
        model.forward(batch)
        model.optimzer_G.zero_grad()
        model.backward_G()
        model.optimzer_G.step()
        model.set_requires_grad()
        model.optimzer_D.zero_grad()
        model.backward_D()
        model.optimzer_D.step()

        if total_iter % 100 == 1:    
            losses = model.get_current_losses()
            learning_rate = model.get_curent_learning_rate()

            summary.add_scalar('loss/G_A_loss', losses["G_A_loss"], total_iter)
            summary.add_scalar('loss/G_B_loss', losses["G_B_loss"], total_iter)
            summary.add_scalar('loss/D_A_loss', losses["D_A_loss"], total_iter)
            summary.add_scalar('loss/D_B_loss', losses["D_B_loss"], total_iter)
            summary.add_scalar('loss/fwd_loss', losses["forward_cycle_loss"], total_iter)
            summary.add_scalar('loss/bkwd_loss', losses["backward_cycle_loss"], total_iter)
            summary.add_scalar('loss/full_loss', losses["full_loss"], total_iter)
            summary.add_scalar('learning_rate', learning_rate, total_iter)
            summary.add_scalars('loss/loss', {"G_A_loss": losses["G_A_loss"],
                                            "G_B_loss": losses["G_B_loss"],
                                            "D_A_loss": losses["D_A_loss"],
                                            "D_B_loss": losses["D_B_loss"],
                                            "fwd_loss": losses["forward_cycle_loss"],
                                            "bkwd_loss": losses["backward_cycle_loss"]}, total_iter)

        if i in [1,2,3,4,5,6,7,8] and epoch % 20 == 0:
            images = model.get_current_images()
            images_list = []

            # ['real_A', 'fake_B', 'regen_A', 'real_B', 'fake_A', 'regen_B']
            for key, value in images.items():
                tmp = (value + 1.0) * 0.5
                images_list += [torch.squeeze(tmp)]

            summary.add_images(f'images/{i}', torch.stack(images_list))

    if epoch % 20 == 0:              
        model.save_networks(epoch)
        
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, 200, time.time() - epoch_start_time))
    model.update_learning_rate()                 
    