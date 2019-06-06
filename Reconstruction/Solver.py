import torch
import time
import numpy as np

class Solver():
    """
    Solver class used to train network with given loss_function & optimizer

    """
    def __init__(self, model, loss_function, optimizer):
        """
        Initialization
        model : the given torch model
        loss_function : given loss function
        optimizer : given optimizer

        """
        super(Solver, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.gpu_avaliable = torch.cuda.is_available()

class ConvSolver(Solver):
    """
    Simple Conv Solver

    """
    def __init__(self, model, loss_function, optimizer):
        """
        Initialization

        """
        super(ConvSolver, self).__init__(model, loss_function, optimizer)

    def validate(self, data_loader):
        """
        Validation of the given validation set data_loader

        """
        self.model.eval()
        results = []
        for idx, (data, label) in enumerate(data_loader):
            # print(data.shape)
            if self.gpu_avaliable:
                data = data.cuda()
            output = self.model(data)
            loss = self.loss_function(output.view_as(data), data)
            loss += self.latent_loss(self.model.mu, self.model.log_sigma)
            if self.gpu_avaliable:
                results.append(loss.cpu().detach().numpy())
            else:
                results.append(loss.detach().numpy())

        mse_loss = np.mean(results)
        self.model.train()

        return mse_loss

    def latent_loss(self, z_mean, log_sigma):
        """
        Latent loss
        """
        mean_sq = z_mean * z_mean

        return 0.5 * torch.mean(mean_sq - log_sigma + torch.exp(log_sigma) - 1)

    def train_and_decode(self, train_dataloader, valid_dataloader,
                         test_dataloader, max_epoch=100, later=0):
        """
        Training the model & get validation and test result

        """
        if self.gpu_avaliable:
            self.model.cuda()
        for i in range(max_epoch):
            start_time = time.time()
            losses = []

            for idx, (img, label) in enumerate(train_dataloader):
                #print(img.data)
                # print(img.shape)
                if self.gpu_avaliable:
                    img = torch.autograd.Variable(img).cuda()
                    # label = torch.autograd.Variable(label).cuda()
                else:
                    img = torch.autograd.Variable(img)
                    # label = torch.autograd.Variable(label)

                # print(img.shape)
                self.optimizer.zero_grad()
                output = self.model(img)
                # print(output.shape)
                loss_1 = self.loss_function(output.view_as(img), img)
                loss_kl = self.latent_loss(self.model.mu, self.model.log_sigma)
                batch_loss = loss_1 + loss_kl
                if self.gpu_avaliable:
                    losses.append(batch_loss.detach().cpu().numpy())
                else:
                    losses.append(batch_loss.detach().numpy())
                #print(losses)
                batch_loss.backward()
                self.optimizer.step()
            #print("LOSSES:",losses)
            epoch_loss = np.mean(losses)

            if i % 10 == 0:
                print('Train:\tEpoch : {}\tTime : {}s\tMSELoss : {}'.format(i, time.time() - start_time, epoch_loss))

            if i < later:
                continue


            start_time = time.time()
            dev_mse_loss = self.validate(valid_dataloader)
            if i % 10 == 0:
                print('Evaluation:\tEpoch : {}\tTime : {}s\tMSELoss : {}'.format(i, time.time() - start_time, dev_mse_loss))
            start_time = time.time()
            test_mse_loss = self.validate(test_dataloader)
            if i % 10 == 0:
                print('Test:\tEpoch : {}\tTime : {}s\tMSELoss : {}\n'.format(i, time.time() - start_time, test_mse_loss))

class LinearSolver(Solver):
    """
    Simple Linear Solver

    """
    def __init__(self, model, loss_function, optimizer, num_channels, conditional=False):
        """
        Initialization

        """
        super(LinearSolver, self).__init__(model, loss_function, optimizer)
        self.conditional = conditional

    def validate(self, data_loader):
        """
        Validation of the given validation set data_loader

        """
        self.model.eval()
        results = []
        for idx, (data, label) in enumerate(data_loader):
            # print(data.shape)
            data = data.view(data.shape[0], -1)
            if not self.conditional:
                output = self.model(data)
            else:
                output = self.model(data, label)
            loss = self.loss_function(output.view_as(data), data)
            loss += self.latent_loss(self.model.mu, self.model.log_sigma)
            results.append(loss.detach().numpy())

        mse_loss = np.mean(results)
        self.model.train()

        return mse_loss

    def latent_loss(self, z_mean, log_sigma):
        """
        Latent loss
        """
        mean_sq = z_mean * z_mean

        return 0.5 * torch.mean(mean_sq - log_sigma + torch.exp(log_sigma) - 1)

    def train_and_decode(self, train_dataloader, valid_dataloader,
                         test_dataloader, max_epoch=100, later=0):
        """
        Training the model & get validation and test result

        """
        if self.gpu_avaliable:
            self.model.cuda()
        for i in range(max_epoch):
            start_time = time.time()
            losses = []

            for idx, (img, label) in enumerate(train_dataloader):
                # print(img.shape)
                if self.gpu_avaliable:
                    img = torch.autograd.Variable(img.view(img.shape[0], -1)).cuda()
                    label = torch.autograd.Variable(label).cuda()
                else:
                    img = torch.autograd.Variable(img.view(img.shape[0], -1))
                    label = torch.autograd.Variable(label)

                self.optimizer.zero_grad()
                if not self.conditional:
                    output = self.model(img)
                else:
                    output = self.model(img, label)
                loss_1 = self.loss_function(output.view_as(img), img)
                loss_kl = self.latent_loss(self.model.mu, self.model.log_sigma)
                batch_loss = loss_1 + loss_kl
                # print(batch_loss)
                if self.gpu_avaliable:
                    losses.append(batch_loss.detach().cpu().numpy())
                else:
                    losses.append(batch_loss.detach().numpy())
                batch_loss.backward()
                self.optimizer.step()

            epoch_loss = np.mean(losses)

            if i % 10 == 0:
                print('Train:\tEpoch : {}\tTime : {}s\tMSELoss : {}'.format(i, time.time() - start_time, epoch_loss))

            if i < later:
                continue

            #if self.gpu_avaliable:
            #    self.model.cpu()
            start_time = time.time()
            dev_mse_loss = self.validate(valid_dataloader)
            if i % 10 == 0:
                print('Evaluation:\tEpoch : {}\tTime : {}s\tMSELoss : {}'.format(i, time.time() - start_time, dev_mse_loss))
            start_time = time.time()
            test_mse_loss = self.validate(test_dataloader)
            if i % 10 == 0:
                print('Test:\tEpoch : {}\tTime : {}s\tMSELoss : {}\n'.format(i, time.time() - start_time, test_mse_loss))
