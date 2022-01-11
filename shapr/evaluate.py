from tqdm import tqdm
import torch
import torch.nn.functional as F
from metrics import SoftDiceLoss

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    error = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['obj']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            #error += dice_coeff(mask_pred, mask_true)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return error
    return error / num_val_batches