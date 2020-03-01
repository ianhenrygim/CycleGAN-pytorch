from parser import CustomParser
from data import get_dataloader
from model import CycleGAN
import time

opt = CustomParser().get_parser()
dataloader = get_dataloader(opt)
model = CycleGAN(opt)
model.setup()

total_iter = 0
epoch_iter = 0

# training
for epoch in range(1, 201):
    epoch_iter += 1

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

        print(time.time() - start_time)
        
    print("epoch : ", epoch_iter)
    