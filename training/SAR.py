import torch
import cv2
import numpy as np
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SARDataset(torch.utils.data.Dataset):
    def __init__(self, files_df, const_norm, transforms=None):
        self.files = files_df
        self.transforms = transforms
        self.const_norm = const_norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = cv2.imread(self.files.image_path[idx], cv2.IMREAD_UNCHANGED)
        mask =  cv2.imread(self.files.mask_path[idx], cv2.IMREAD_UNCHANGED)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # convert image from 0-255 scale to 0-1 scale and normalize with imagenet mean & std
            normalize = A.Compose([self.const_norm])
            normalized = normalize(image=image, mask=mask)
            image = normalized['image']

        sample = dict(image=image, mask=mask)

        # convert array dimensions from HWC to CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)

        return sample

class SARDataModule(pl.LightningDataModule):
    def __init__(self, dataset_df, const_norm, nodata_color, image_size=640, batch_size=4, data_split=dict(train=0.72, valid=0.14, test=0.14), image_width=None, image_height=None, divisor=1):
        super().__init__()
        self.save_hyperparameters()
        self.const_norm = const_norm

        if self.hparams.dataset_df.split_name.isna().sum() == len(self.hparams.dataset_df):
            self.train_df, self.valid_df, self.test_df = np.split(self.hparams.dataset_df.sample(frac=1, random_state=np.random.RandomState(seed=42)),
                                                                [int(self.hparams.data_split['train'] * len(self.hparams.dataset_df)),
                                                                int((self.hparams.data_split['train'] + self.hparams.data_split['valid']) * len(self.hparams.dataset_df))
                                                                ])
        else:
            self.train_df = self.hparams.dataset_df[(self.hparams.dataset_df.split_name=='TRAIN')]
            self.valid_df = self.hparams.dataset_df[(self.hparams.dataset_df.split_name=='VALID')]
            self.test_df = self.hparams.dataset_df[(self.hparams.dataset_df.split_name=='TEST')]

        if image_width is None:
          image_width = image_size
        if image_height is None:
          image_height = image_size

        image_width_padded = image_width
        if image_width % divisor != 0:
          image_width_padded += divisor - image_width % divisor

        image_height_padded = image_height
        if image_height % divisor != 0:
          image_height_padded += divisor - image_height % divisor


        self.train_transforms = A.Compose([
                                        #    A.Resize(height=image_height, width=image_width, p=1),
                                        #    A.augmentations.geometric.transforms.PadIfNeeded(
                                        #        min_height=None,
                                        #        min_width=None,
                                        #        pad_height_divisor=divisor,
                                        #        pad_width_divisor=divisor
                                        #        ),
                                        #    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                                        #    A.UnsharpMask(p=0.3),
                                           A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.4, rotate_limit=45,
                                                              border_mode=cv2.BORDER_CONSTANT,
                                                              value=nodata_color, mask_value=0, p=0.5),
                                           A.HorizontalFlip(p=0.5),
                                           A.VerticalFlip(p=0.5),
                                           const_norm
                                           ])

        self.valid_transforms = A.Compose([
                                        #    A.Resize(height=image_height, width=image_width, p=1),
                                        #    A.augmentations.geometric.transforms.PadIfNeeded(
                                        #        min_height=None,
                                        #        min_width=None,
                                        #        pad_height_divisor=divisor,
                                        #        pad_width_divisor=divisor
                                        #        ),
                                           const_norm
                                           ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SARDataset(files_df=self.train_df.reset_index(drop=True), const_norm = self.const_norm, transforms=self.train_transforms)
            self.valid_dataset = SARDataset(files_df=self.valid_df.reset_index(drop=True), const_norm = self.const_norm, transforms=self.valid_transforms)

        if stage == 'test' or stage is None:
            self.test_dataset = SARDataset(files_df=self.test_df.reset_index(drop=True), const_norm = self.const_norm, transforms=self.valid_transforms)

        if stage == 'predict' or stage is None:
            self.predict_dataset = SARDataset(files_df=self.hparams.dataset_df.reset_index(drop=True), const_norm = self.const_norm, transforms=self.valid_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, shuffle=False, batch_size=self.hparams.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.hparams.batch_size, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, shuffle=False, batch_size=self.hparams.batch_size, drop_last=True)


import torch.nn.functional as F
def focal_loss_with_logits(output, target, gamma=2.0, alpha=0.25, reduction='mean', normalized=False, reduced_threshold=None, eps=1e-7):
    # compute the binary cross entropy term
    bce_loss = F.binary_cross_entropy_with_logits(output, target.float(), reduction='none')

    # compute the probability of being classified as positive
    pt = torch.exp(-bce_loss)

    # handle the NoneType error
    if pt is None:
        pt = eps

    # compute the focal term
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)

    # compute the focal loss
    focal_loss = alpha * focal_term * bce_loss

    # apply reduction
    if reduction == 'mean':
        if normalized:
            num_positives = torch.sum(target > 0.5)
            focal_loss = torch.sum(focal_loss) / (num_positives + eps)
        else:
            focal_loss = torch.mean(focal_loss)
    elif reduction == 'sum':
        focal_loss = torch.sum(focal_loss)

    return focal_loss

class SARModel(pl.LightningModule):

    def __init__(self, hparams_, **kwargs):
        self.hparams_ = hparams_
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # read params passed to the model and use them
        self.learning_rate = hparams_.learning_rate
        self.model = smp.create_model(arch=hparams_.arch_name,
                                      encoder_name=hparams_.encoder_name,
                                      encoder_weights=hparams_.encoder_weights,
                                      # encoder_depth=5,
                                      # psp_out_channels=640,
                                      # upsampling=32,
                                      # decoder_attention_type=hparams_.decoder_attention_type,
                                      in_channels=hparams_.in_channels,
                                      classes=hparams_.out_classes, **kwargs)

        # self.model = UNet(input_shape=hparams_.input_shape,
        #                   IMG_CLASSES=hparams_.out_classes)

        if hasattr(hparams_, 'load_weights_from') and hparams_.load_weights_from is not None:
            self.model = SARModel.load_from_checkpoint(hparams_.load_weights_from)

        # for image segmentation dice loss could be the best first choice; but other options can also be used
        if hparams_.loss_function == 'DICE':
            self.loss_fn = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        elif hparams_.loss_function == 'BCE':
            self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        elif hparams_.loss_function == 'FOCAL':
            self.loss_fn = focal_loss_with_logits#smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE, alpha=hparams_.loss_alpha, gamma=hparams_.loss_gamma)
        elif hparams_.loss_function == 'JACCARD':
            self.loss_fn = smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        elif hparams_.loss_function == 'LOVASZ':
            self.loss_fn = smp.losses.LovaszLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        elif hparams_.loss_function == 'TVERSKY':
            self.loss_fn = smp.losses.TverskyLoss(mode=smp.losses.BINARY_MODE, alpha=hparams_.loss_alpha, beta=hparams_.loss_beta, gamma=hparams_.loss_gamma, from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        self.log(f"{stage}_loss", loss)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        per_image_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_per_image_f1": per_image_f1,
            f"{stage}_dataset_f1": dataset_f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def train_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        epoch_average = self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return epoch_average

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        self.test_step_outputs.append(loss)
        return loss

    def on_test_epoch_end(self):
        epoch_average = self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()
        return epoch_average

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=3, factor=0.1)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": 'valid_loss',
                                 "interval": 'epoch',
                                 "frequency": 1
                                }
               }