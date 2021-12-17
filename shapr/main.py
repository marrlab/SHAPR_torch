from shapr.utils import *
from shapr import settings
from shapr.data_generator import *
from shapr.model import netSHAPR, netDiscriminator
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from evaluate import evaluate
from pathlib import Path


PARAMS = {"num_filters": 10,
      "dropout": 0.
}

"""
Set the path where the following folders are located: 
- obj: containing the 3D groundtruth segmentations 
- mask: containg the 2D masks 
- image: containing the images from which the 2D masks were segmented (e.g. brightfield)
All input data is expected to have the same x and y dimensions and the obj (3D segmentations to have a z-dimension of 64.
The filenames of corresponding files in the obj, mask and image ordner are expeted to match.
"""


def run_train(amp: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(settings)
    """
    Get the filenames
    """
    filenames = os.listdir(os.path.join(settings.path, "obj/"))

    """
    We train the model on all data on 5 folds, while the folds are randomly split
    """
    kf = KFold(n_splits=5)
    os.makedirs(os.path.join(settings.path, "logs"), exist_ok=True)

    for fold, (cv_train_indices, cv_test_indices) in enumerate(kf.split(filenames)):
        cv_train_filenames = [str(filenames[i]) for i in cv_train_indices]
        cv_test_filenames = [str(filenames[i]) for i in cv_test_indices]

        """
        From the train set we use 20% of the files as validation during training 
        """
        dataset = SHAPRDataset(settings.path, cv_train_filenames)
        n_val = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        print(f"For Validation we use: {str(n_val)} randomly sampled files")
        print(f"For training we use: {str(n_train)} randomly sampled files")
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        """
        If pretrained weights should be used, please add them here:
        These weights will be used for all folds
        """

        lr = 0.0002
        beta1 = 0.5
        optimizerSHAPR = optim.Adam(netSHAPR.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerDisc = optim.Adam(netDiscriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        criterionSHAPR = nn.BCELoss()
        criterionDisc = nn.BCELoss()
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizerSHAPR, 'max', patience=2)  # goal: maximize Dice score

        loader_args = dict(batch_size=settings.batch_size, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        global_step = 0
        for epoch in range(settings.epochs_SHAPR):
            netSHAPR.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{settings.epochs_SHAPR}', unit='img') as pbar:
                for batch in train_loader:
                    images = batch['image']
                    true_obj = batch['obj']
                    images = images.to(device=device, dtype=torch.float32)
                    true_obj = true_obj.to(device=device, dtype=torch.float32)

                    with torch.cuda.amp.autocast(enabled=amp):
                        obj_pred = netSHAPR(images)
                        loss = criterionSHAPR(obj_pred, true_obj)

                    optimizerSHAPR.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizerSHAPR)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    #experiment.log({
                    #    'train loss': loss.item(),
                    #    'step': global_step,
                    #    'epoch': epoch
                    #})
                    #pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    division_step = (n_train // (10 * settings.batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            #histograms = {}
                            #for tag, value in netSHAPR.named_parameters():
                            #    tag = tag.replace('/', '.')
                            #    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            #    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            val_score = evaluate(netSHAPR, val_loader, device)
                            scheduler.step(val_score)

                            #logging.info('Validation Dice score: {}'.format(val_score))
                            #experiment.log({
                            #    'learning rate': optimizer.param_groups[0]['lr'],
                            #    'validation Dice': val_score,
                            #    'images': wandb.Image(images[0].cpu()),
                            #   'masks': {
                            #        'true': wandb.Image(true_masks[0].float().cpu()),
                            #        'pred': wandb.Image(
                            #            torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            #    },
                            #    'step': global_step,
                            #    'epoch': epoch,
                            #    **histograms
                            #})
            Path(settings.path+ "/logs/").mkdir(parents=True, exist_ok=True)
            torch.save(netSHAPR.state_dict(), str(settings.path+ "/logs/" + 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            #logging.info(f'Checkpoint {epoch + 1} saved!')

        """
        After training SHAPR for the set number of epochs, we train the adverserial model
        """

        real_label = 1.
        fake_label = 0.
        for epoch in range(settings.epochs_cSHAPR):
            netSHAPR.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{settings.epochs_SHAPR}', unit='img') as pbar:
                for batch in train_loader:
                    images = batch['image']
                    true_obj = batch['obj']
                    images = images.to(device=device, dtype=torch.float32)
                    true_obj = true_obj.to(device=device, dtype=torch.float32)

                    images = images.to(device=device, dtype=torch.float32)
                    true_obj = true_obj.to(device=device, dtype=torch.float32)

                    # (1) Update Discriminator network
                    ## Train with all-real batch

                    netDiscriminator.zero_grad()
                    # Format batch
                    b_size = images.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    # Forward pass real batch through D
                    output = netDiscriminator(true_obj).view(-1)
                    # Calculate loss on all-real batch
                    errD_real = criterionDisc(output, label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = output.mean().item()

                    ## Train with all-fake batch

                    fake = netSHAPR(images)
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    output = netDiscriminator(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterionDisc(output, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # Update D
                    optimizerDisc.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    netSHAPR.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = netDiscriminator(fake).view(-1)
                    # Calculate G's loss based on this output
                    errSHAPR = criterionSHAPR(output, label)
                    # Calculate gradients for G
                    errSHAPR.backward()
                    D_G_z2 = output.mean().item()
                    # Update G
                    optimizerSHAPR.step()
                    division_step = (n_train // (10 * settings.batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            # histograms = {}
                            # for tag, value in netSHAPR.named_parameters():
                            #    tag = tag.replace('/', '.')
                            #    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            #    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            val_score = evaluate(netSHAPR, val_loader, device)
                            scheduler.step(val_score)
            Path(settings.path + "/logs/").mkdir(parents=True, exist_ok=True)
            torch.save(netSHAPR.state_dict(), str(settings.path + "/logs/" + 'Adverserial_model_checkpoint_epoch{}.pth'.format(epoch + 1)))

        """
        The 3D shape of the test data for each fold will be predicted here
        """
        for test_file in cv_test_filenames:
            image = torch.from_numpy(get_test_image(settings, test_file))

            netSHAPR.eval()
            img = image.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                output = netSHAPR(img)

            """
            The predictions on the test set for each fold will be saved to the results folder
            """
            prediction = output * 255
            os.makedirs(settings.result_path, exist_ok=True)
            imsave(os.path.join(settings.result_path, test_file), prediction.astype("uint8"))


def run_evaluation(): 

    print(settings)

    #TODO

    """
    Get the filenames
    """
    '''test_filenames = os.listdir(os.path.join(settings.path, "obj"))

    model2D = netSHAPR(PARAMS)
    model2D.load_weights(settings.pretrained_weights_path)

    """
    If pretrained weights should be used, please add them here:
    These weights will be used for all folds
    """

    """
    The 3D shape of the test data for each fold will be predicted here
    """
    test_data = data_generator_test_set(settings.path, test_filenames)

    predict = model2D.predict_generator(test_data, steps = len(test_filenames))
    print(np.shape(predict))

    """
    The predictions on the test set for each fold will be saved to the results folder
    """
    #save predictions
    print(np.shape(predict))
    i = 0
    for i, test_filename in enumerate(test_filenames):
        result = predict[i,...]*255
        os.makedirs(settings.result_path, exist_ok=True)
        imsave(settings.result_path + test_filename, result.astype("uint8"))
        i = i+1        
    '''


