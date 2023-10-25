
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch_fidelity

class ImageList(torch.utils.data.Dataset):
    def __init__(self, src):
        self.src=src
        image_size=384
        self.transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.PILToTensor(),
    ])
    def __getitem__(self, idx):
        img=self.src[idx]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        return img
    def __len__(self,):
        return len(self.src)


def fidelity_on_dataset(src, dst, **kwargs):
    defaults=dict(
        isc=True,
        fid=True,
        verbose=False,
    )
    defaults.update(kwargs)
    fidelity = torch_fidelity.calculate_metrics(
        input1=ImageList(src), 
        input2=ImageList(dst),
        **defaults,
        )
    return fidelity

def fidelity_on_image(src, dst, **kwargs):
    defaults=dict(
        isc=False,
        fid=True,
        verbose=False,
    )
    defaults.update(kwargs)
    return fidelity_on_dataset([src], [dst], **defaults)
