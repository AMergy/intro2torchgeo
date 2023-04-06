#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   resnet50_trainer.py
@Time    :   2023/03/31 11:48:12
@Author  :   AMergy 
@Version :   1.0
@Contact :   anne.mergy@gmail.com
@Desc    :   Script to retrain first and last layers of ResNet50
'''
import os
import argparse
import tempfile
from torchgeo.datasets import NAIP, ChesapeakeDE
from torchgeo.datasets.utils import download_url, stack_samples
from torchgeo.models import FarSeg
from torchgeo.samplers import RandomGeoSampler
import torch
from torch.utils.data import DataLoader

TILES = [
    "m_3807511_ne_18_060_20181104.tif",
    "m_3807511_se_18_060_20181104.tif",
    "m_3807512_nw_18_060_20180815.tif"]

class TorchGeoTL:
    def __init__(self, args):
        """
        Initialize the class TorchGeoTL to train and test the model
        Args:
            args: argparse object
        """
        self.model_name = args.model_name
        self.args = args
        self.out_dir = args.out_dir

        if args.run_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
                os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
                print("Running on GPU")
            else:
                print("No GPU available - running on cpu")
        else:
            self.device = torch.device("cpu")
            # Set the number of threads to be used by PyTorch
            torch.set_num_threads(args.num_workers)
            # Set the number of CPUs to be used by the Python script
            os.environ["OMP_NUM_THREADS"] = str(args.num_workers)
            os.environ["MKL_NUM_THREADS"] = str(args.num_workers)
            os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_workers)
            print(f"Running on {args.num_workers} cpus")

        self.dl_train, self.dl_val, self.dl_test = self._setup_dataset()

        self.error_metrics = self._setup_metrics()


    def _setup_metrics(self):
        """Set up metrics

        Returns:
            dict: dictionary of metrics
        """
        error_metrics = {'CrossEntropy': torch.nn.CrossEntropyLoss()}

        print('error_metrics:', error_metrics.keys())
        error_metrics['loss'] = error_metrics[self.args.loss_key]
        return error_metrics

    def _setup_optimizer(self, model):
        """Returns an optimizer based on the optimizer specified in parser

        Args:
            model (torchgeo.models.FarSeg): model to optimize

        Raises:
            ValueError: if optimizer is not ADAM or SGD

        Returns:
            torch.optim: optimizer
        """
        if self.args.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=self.args.base_learning_rate,
                                         weight_decay=self.args.l2_lambda)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=self.args.base_learning_rate,
                                        weight_decay=self.args.l2_lambda)
        else:
            raise ValueError("Solver '{}' is not defined.".format(self.args.optimizer))
        return optimizer

    def _setup_dataset(self):
        """Splits dataset into training, validation and test sets

        Returns:
            torch.utils.data.DataLoader: dataloaders for train, validation and test
        """
        data_root = tempfile.gettempdir()
        naip_url = (
            "https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
        )
        for i, tmp_folder in enumerate(["naip_train","naip_val","naip_test"]):
            naip_root = os.path.join(data_root, tmp_folder)
            download_url(naip_url + TILES[i], naip_root)

        chesapeake_root = os.path.join(data_root, "chesapeake_train")
        chesapeake = ChesapeakeDE(chesapeake_root, download=True)
        
        # Split dataset into three subsets
        train_naip = NAIP(os.path.join(data_root, "naip_train"),
                          crs=chesapeake.crs,
                          res=chesapeake.res,
                          cache=True)
        train_chesapeake = ChesapeakeDE(chesapeake_root, cache=True)
        train_dataset = train_chesapeake & train_naip
        train_sampler = RandomGeoSampler(train_dataset, size=256, length=10000)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, sampler=train_sampler, 
            collate_fn=stack_samples, 
        )
        val_naip = NAIP(os.path.join(data_root, "naip_val"), crs=chesapeake.crs, 
                        res=chesapeake.res, cache=True)
        val_chesapeake = ChesapeakeDE(chesapeake_root, cache=True)
        val_dataset = val_chesapeake & val_naip
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.args.batch_size,
            collate_fn=stack_samples, shuffle=False,
        )
        test_naip = NAIP(os.path.join(data_root, "naip_test"),
                        crs=chesapeake.crs,
                        res=chesapeake.res,
                        cache=True)
        test_chesapeake = ChesapeakeDE(chesapeake_root, cache=True)
        test_dataset = test_chesapeake & test_naip
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.batch_size,
            collate_fn=stack_samples, shuffle=False,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def train(self):
        """A routine to train and validated the model for several epochs.
        """
        # Load the pre-trained ResNet50 model
        # Modify the last layer; there are 12 classes + 0 standing for no_data
        model = FarSeg(backbone=self.model_name, classes=13, backbone_pretrained=True)

        # Freeze all layers except the first and last
        for name, param in model.named_parameters():
            if not ('conv1' in name or 'fc' in name):
                param.requires_grad = False

        # Modify the first layer to accept a 4-band image
        model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=(3, 3), bias=False)

        optimizer = self._setup_optimizer(model)
        criterion = self.error_metrics["loss"]

        model.to(self.device)
        criterion.to(self.device)

        # Train the model
        for epoch in range(self.args.epochs):
            print("training...")
            model.train()
            for i, sample in enumerate(self.dl_train):
                inputs = sample["image"].float()
                labels = sample["mask"]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs.view(outputs.shape[0],
                                            outputs.shape[1],-1), 
                                labels.view(labels.shape[0],-1), 
                                )
                loss.backward()
                optimizer.step()
            print("Epoch {} - Train Loss: {:.4f}".format(epoch, loss.item()))

            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sample in self.dl_val:
                    inputs = sample["image"].float()
                    labels = sample["mask"]
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print("Validation Accuracy: {:.2f}%".format(100 * correct / total))

        # save model weights after last epoch:
        path = os.path.join(self.out_dir,'weights_{}_epochs.pt'.format(self.args.epochs))
        torch.save(self.model.state_dict(), path)
        print('Saved weights at {}'.format(path))

    def test(self, model_weights_path=None, dl_test=None):
        """Test trained model on test data.

        Args:
            model_weights_path (_type_, optional): path to trained model weights.
                                                   If None, loads "best_weights.pt".
                                                   Defaults to None.
            dl_test (_type_, optional): torch dataloader with test data. If None,
                                        self.dl_test is loaded. Defaults to None.

        Returns:
            int: number of correct predictions
            int: total number of predictions considered for the evaluation
        """
        if dl_test is None:
            dl_test = self.dl_test
        # test performance

        if model_weights_path is None:
            model_weights_path = os.path.join(self.out_dir, 'best_weights.pt')

        # load best model weights
        model = FarSeg(backbone='resnet50', classes=13, backbone_pretrained=False)
        model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=(3, 3), bias=False)
        model.load_state_dict(torch.load(model_weights_path))

        # Set the model to evaluation mode
        model.eval()

        # Evaluate on test set
        correct = 0
        total = 0
        with torch.no_grad():
            for sample in self.dl_test:
                inputs = sample["image"].float()
                labels = sample["mask"]
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Test Accuracy: {:.2f}%".format(100 * correct / total))

        return correct, total


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        help="Number of epochs to re-train the first and last layer",
        default=5)
    parser.add_argument(
        "--batch_size",
        help="Batch size",
        default=64)
    parser.add_argument(
        "--optimizer",
        help="Optimizer to use for the training",
        default="ADAM")
    parser.add_argument(
        "--l2_lambda",
        help="Decay rate of the weights",
        default=1e-5)
    parser.add_argument(
        "--base_learning_rate",
        help="Learning rate",
        default=1e-4)
    parser.add_argument(
        "--loss_key",
        help="Loss function",
        default="CrossEntropy")
    parser.add_argument(
        "--num_workers",
        help="Max number of cpus",
        default=4)
    parser.add_argument(
        "--run_gpu",
        help="Whether to use a GPU",
        default=False)
    parser.add_argument(
        "--out_dir",
        help="Where to save the results",
        default="res/")
    parser.add_argument(
        "--model_name",
        help="Model name",
        default="resnet50")
    args = parser.parse_args()

    # Initialize trainer
    transfert_learning = TorchGeoTL(args)

    # Train model
    transfert_learning.train()

    path = os.path.join(args.out_dir,'weights_{}_epochs.pt'.format(args.epochs))

    # Test model
    transfert_learning.test(model_weights_path=path)
