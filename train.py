from parser import CustomParser
from data import get_dataloader
from model import CycleGAN
from tqdm import tqdm
import time

opt = CustomParser().get_parser()
dataloader = get_dataloader(opt)
model = CycleGAN(opt)
model.setup()

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

        if total_iter % 200 == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            
            # ['D_A_loss', 'D_B_loss', 'G_A_loss', 'G_B_loss', 'forward_cycle_loss', 'backward_cycle_loss'
            loss_result = f'D_A : {losses["D_A_loss"]:<10}|D_B : {losses["D_B_loss"]:<10}|G_A : {losses["G_A_loss"]:<10}|G_B : {losses["G_B_loss"]:<10}|forward : {losses["forward_cycle_loss"]:<10}|backware : {losses["backward_cycle_loss"]:<10}|'
            print(f'Total iter : {total_iter}')
            print(loss_result)

    if epoch % 10 == 0:              # cache our model every <save_epoch_freq> epochs
        model.save_networks(epoch)
        
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, 200, time.time() - epoch_start_time))
    model.update_learning_rate()                 
    