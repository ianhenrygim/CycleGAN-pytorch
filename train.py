from parser import CustomParser
from data import get_dataloader

opt = CustomParser().get_parser()
dataloader = get_dataloader(opt)

# test
for i, batch in enumerate(dataloader):
    print(i, batch)

    if (i == 10):
        break