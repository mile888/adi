######## Model Training ###########

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A

from model import UNet, DeepLab
from dataset import CTDataset, ANNOTATION_PATH
import loss



class SegModel(pl.LightningModule):
    def __init__(
        self,
        transforms,
        optimizer: str,
        scheduler: str,
        lr: float,
        batch_size: int,
    ):
        super(SegModel, self).__init__()
        self.num_classes = 3

        self.net = UNet(self.num_classes)

        self.train_dataset = CTDataset(ANNOTATION_PATH, transforms)
        self.test_dataset = CTDataset(ANNOTATION_PATH, transforms)

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.eps = 1e-7

        # Visualization
        self.color_map = torch.FloatTensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

        
    def forward(self, x):
        return self.net(x)

    
    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img)

        loss = F.cross_entropy(pred, mask)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img)

        intersection, union, target = loss.calc_val_data(pred, mask, self.num_classes)

        return {'intersection': intersection, 'union': union, 'target': target, 'img': img, 'pred': pred, 'mask': mask}

    
    def validation_epoch_end(self, outputs):
        intersection = torch.cat([x['intersection'] for x in outputs])
        union = torch.cat([x['union'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs])

        mean_iou, mean_class_acc, mean_acc = loss.calc_val_loss(intersection, union, target, self.eps)

        log_dict = {'mean_iou': mean_iou, 'mean_class_acc': mean_class_acc, 'mean_acc': mean_acc}

        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True)

        # Visualize results
        img = torch.cat([x['img'] for x in outputs]).cpu()
        pred = torch.cat([x['pred'] for x in outputs]).cpu()
        mask = torch.cat([x['mask'] for x in outputs]).cpu()

        pred_vis = self.visualize_mask(torch.argmax(pred, dim=1))
        mask_vis = self.visualize_mask(mask)

        img = torch.cat([img, img, img], dim=1)
        results = torch.cat(torch.cat([img, pred_vis, mask_vis], dim=3).split(1, dim=0), dim=2)
        results_thumbnail = F.interpolate(results, scale_factor=0.25, mode='bilinear')[0]

        self.logger.experiment.add_image('results', results_thumbnail, self.current_epoch)

        
    def visualize_mask(self, mask):
        b, h, w = mask.shape
        mask_ = mask.view(-1).to(dtype=torch.long)

        if self.color_map.device != mask.device:
            self.color_map = self.color_map.to(mask.device)

        mask_vis = self.color_map[mask_].view(b, h, w, 3).permute(0, 3, 1, 2).clone()

        return mask_vis

    
    def configure_optimizers(self):
        optim = {'adam': torch.optim.Adam(self.net.parameters(), lr= 1e-3, weight_decay=1e-6)}
        scheduler = {'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(optim[self.optimizer], 10)}
                       
        opt = optim[self.optimizer]
        sch = scheduler[self.scheduler]
        
        return [opt], [sch]

    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=4, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=4, batch_size=self.batch_size, shuffle=False)


    
if __name__ == "__main__":
    
    transform = A.Compose([
        A.Normalize(20142.5019, 285.7593, max_pixel_value=1),
        A.RandomResizedCrop(255, 255),
        A.RandomRotate90()
        ])
    
    model = SegModel(transform, "adam", "CosineAnnealingLR", 1e-3, 18)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)