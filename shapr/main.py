from shapr._settings import SHAPRConfig
from shapr.data_generator import get_test_image
from shapr.metrics import Dice_loss, IoU_error, Volume_error

from model import LightningSHAPRoptimization, LightningSHAPR_GANoptimization

from skimage.io import imsave

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import numpy as np

import os
import torch
import wandb

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


def run_train(amp: bool = False, params=None, overrides=None, args=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings = SHAPRConfig(params=params)

    if overrides is not None:
        # Handle overrides by user
        for k, v in overrides.items():
            settings.__setattr__(k, v)

    # Handle GPU vs CPU selection
    if device == torch.device("cpu"):
        gpus = None
    else:
        gpus = 1

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

    items = [
        (k, v) for k, v in settings.__dict__.items()
        if not k.startswith('_')
    ]

    # Prepare configuration for `wandb` client. Additional values can be
    # added on a per-fold basis.
    config = {
        k: v for k, v in items
    }

    # Get the current sweep ID or generate a new one if it does not
    # exist. That way, everything is grouped correctly via `wandb`.
    group = os.getenv('WANDB_SWEEP_ID', default=wandb.util.generate_id())

    # Get all folds already, making it easier to run them later on since
    # we are not dealing with a generator expression any more.
    folds = [
        (fold, train_index, test_index)
        for fold, (train_index, test_index) in enumerate(kf.split(filenames))
    ]

    if args is not None and args.fold is not None:
        fold = args.fold

        run_fold(
            config,
            group=group,
            fold=fold,
            train_index=folds[fold][1],
            test_index=folds[fold][2],
            filenames=filenames,
            settings=settings,
            gpus=gpus,
        )
    else:
        for fold, train_index, test_index in folds:
            run_fold(
                config,
                group=group,
                fold=fold,
                train_index=train_index,
                test_index=test_index,
                filenames=filenames,
                settings=settings,
                project='SHAPR_topological',
                entity='shapr_topological',
                gpus=gpus,
            )


def run_fold(
    config,
    group,
    fold,
    train_index,
    test_index,
    filenames,
    settings,
    project=None,
    entity=None,
    reinit=False,
    job_type='train',
    gpus=None,
):
    """Runs an individual fold.

    This function is either called in a sweep setting to process
    a single fold or it will process all folds.

    Parameters
    ----------
    config : dict
        Overall configuration for `wandb` client.

    group : str
        Group of the individual fold

    fold : int
        Specifies which fold the run pertains to.

    train_index : array_like
        Train indices for fold

    test_index : array_like
        Test indices for fold

    filenames : array_like
        Filenames for cross-validation

    settings : SHAPRConfig
        Overall settings of run

    project : str or `None`
        Project for `wandb` training

    entity : str or `None`
        Overall entity for `wandb` training

    reinit : bool
        If set, run will be re-initialised. This only makes sense for
        non-sweep runs. The behaviour during sweeps is brittle.

    job_type : str
        Type of job

    gpus : int or `None`
        Specifies which GPUs to use
    """
    config['fold'] = fold
    wandb_logger = WandbLogger()

    run = wandb.init(
        project=project,
        entity=entity,
        job_type=job_type,
        group=group,
        config=config,
        reinit=reinit,
    )

    cv_train_filenames = [str(filenames[i]) for i in train_index]
    cv_test_filenames = [str(filenames[i]) for i in test_index]

    # From the train set we use 20% of the files as validation during training
    cv_train_filenames, cv_val_filenames = train_test_split(
        cv_train_filenames, test_size=0.2
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/combined_loss",
        dirpath=os.path.join(settings.path, "logs"),
        filename="SHAPR_training-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor='val/combined_loss', patience=15
    )

    SHAPRmodel = LightningSHAPRoptimization(
        settings,
        cv_train_filenames,
        cv_val_filenames,
        cv_test_filenames
    )

    SHAPR_trainer = pl.Trainer(
        max_epochs=settings.epochs_SHAPR,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[wandb_logger],
        log_every_n_steps=5,
        gpus=gpus
    )
    SHAPR_trainer.fit(model= SHAPRmodel)
    torch.save({
        'state_dict': SHAPRmodel.state_dict(),
    }, os.path.join(settings.path, "logs/")+"SHAPR_training.ckpt")

    if settings.epochs_SHAPR > 0:
        SHAPR_best_model_path = checkpoint_callback.best_model_path
    else:
        SHAPR_best_model_path = None

    # After training SHAPR for the set number of epochs, we train the
    # adversarial model
    early_stopping_callback = EarlyStopping(
        monitor='val/combined_loss', patience=15
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/combined_loss",
        dirpath=os.path.join(settings.path, "logs"),
        verbose=True,
        filename="SHAPR_GAN_training-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    SHAPR_GANmodel = LightningSHAPR_GANoptimization(
        settings,
        cv_train_filenames,
        cv_val_filenames,
        cv_test_filenames,
        SHAPR_best_model_path
    )

    SHAPR_GAN_trainer = pl.Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=settings.epochs_cSHAPR,
        logger=[wandb_logger],
        gpus=gpus
    )
    SHAPR_GAN_trainer.fit(model=SHAPR_GANmodel)

    """
    The 3D shape of the test data for each fold will be predicted here
    """
    if settings.epochs_cSHAPR > 0:
        SHAPR_GAN_trainer.test(model=SHAPR_GANmodel)

        if len(settings.result_path) > 0:
            with torch.no_grad():
                SHAPR_GANmodel.eval()
                for test_file in cv_test_filenames:
                    image, gt = get_test_image(settings, test_file)
                    image = torch.from_numpy(image)
                    img = image.float()
                    output = SHAPRmodel(img)
                    output = output.squeeze()
                    os.makedirs(settings.result_path, exist_ok=True)
                    prediction = output.cpu().detach().numpy()
                    imsave(os.path.join(settings.result_path, test_file), (255 * prediction).astype("uint8"))
    else:
        SHAPR_trainer.test(model= SHAPRmodel)

        if len(settings.result_path) > 0:
            with torch.no_grad():
                SHAPRmodel.eval()
                for test_file in cv_test_filenames:
                    image, gt = get_test_image(settings, test_file)
                    image = torch.from_numpy(image)
                    img = image.float()
                    output = SHAPRmodel(img)
                    output = output.squeeze()
                    os.makedirs(settings.result_path, exist_ok=True)
                    prediction = output.cpu().detach().numpy()
                    imsave(os.path.join(settings.result_path, test_file), (255 * prediction).astype("uint8"))

    # Finish current `wandb` run; this enables grouping later on.
    run.finish()


def run_evaluation():

    settings = SHAPRConfig()
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
