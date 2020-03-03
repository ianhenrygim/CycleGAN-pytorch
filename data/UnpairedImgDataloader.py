import glob, os, random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def get_dataloader(opt):
    return DataLoader(CustomUnalignedDataset(opt), 
        batch_size=opt.batch_size, 
        shuffle=True if opt.isTrain else False,
        num_workers=2 if opt.isTrain else 0)

class CustomUnalignedDataset(Dataset):
    def __init__(self, opt):
        self.mode = "train" if opt.isTrain else "test"

        # Todo : checking whether all files are image format or not
        print(f"Dataset path : {os.path.join(opt.dataroot, '%sA(andB)' % self.mode)}")
        self.domain_A = sorted(glob.glob(os.path.join(opt.dataroot, '%sA' % self.mode) + '/*.*'))
        self.domain_B = sorted(glob.glob(os.path.join(opt.dataroot, '%sB' % self.mode) + '/*.*'))

        self.transforms = transforms.Compose([
            transforms.Resize(int(opt.img_size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(opt.img_size),
            transforms.RandomHorizontalFlip(0.5 if opt.isTrain else 0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        return {
            'A': self.transforms(
                Image.open(self.domain_A[index % len(self.domain_A)]).convert('RGB')),
            'B': self.transforms(
                Image.open(self.domain_B[random.randint(0, len(self.domain_B) - 1)]).convert('RGB'))}

    def __len__(self):
        return max(len(self.domain_A), len(self.domain_B))
