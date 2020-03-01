import random
import torch

class ImageBuffer():
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.images = []

    def query(self, images):
        if self.buffer_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if len(self.images) < self.buffer_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.images.append(image)
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.buffer_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return torch.cat(return_images, 0)   # collect all the images and return
