import numpy
import torch
import math
import os
import time
import copy
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt

from models.LDSTGNN import LDSTGNN
from utils.logger import get_logger
from utils.metrics import All_Metrics
import nni


def print_model_parameters(model, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
    return total_num


class Trainer(object):
    def __init__(self, model, model_name, loss, optimizer, train_loader, val_loader,
                 test_loader, scaler, lr_scheduler, device, log_dir, grad_norm,
                 max_grad_norm, log_step, lr_decay, epochs, early_stop, early_stop_patience, mae_thresh, mape_thresh,
                 dataset, decay_interval, decay_r, init_r, final_r, l, use_curriculum_learning=True, step_size=5,
                 args=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader)
        self.log_dir = log_dir
        self.best_path = os.path.join(log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(log_dir, 'loss.png')
        # log
        if os.path.isdir(log_dir) is False:
            os.makedirs(log_dir, exist_ok=True)
        self.logger = get_logger(log_dir, name=model_name, debug=False)
        self.logger.info('Experiment log path in: {}'.format(log_dir))
        if args is not None:
            self.logger.info(args)
        total_param = print_model_parameters(model, only_num=False)
        self.logger.info(self.model)
        self.logger.info("Total params: {}".format(str(total_param)))
        self.device = device
        self.grad_norm = grad_norm
        self.max_grad_norm = max_grad_norm
        self.log_step = log_step
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.mae_thresh = mae_thresh
        self.mape_thresh = mape_thresh
        self.dataset = dataset
        self.decay_interval = decay_interval
        self.decay_r = decay_r
        self.init_r = init_r
        self.final_r = final_r
        self.l = l
        self.use_curriculum_learning = use_curriculum_learning
        self.step_size = step_size
        self.args = args

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model(x)
                loss = self.loss(output, y)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(self.val_loader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch, batches_seen, task_level=None):
        self.model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            x = x.to(self.device)
            y = y.to(self.device)
            output, structure_kl_loss_sum = self.model(x, decay_interval=self.decay_interval, decay_r=self.decay_r,
                                                       current_epoch=epoch, init_r=self.init_r, final_r=self.final_r)
            if self.use_curriculum_learning and task_level is not None:
                loss = self.loss(output[:, :task_level, ...], y[:, :task_level, ...])
            else:
                loss = self.loss(output, y)
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
            total_loss += loss.item()
            loss = loss + self.l * structure_kl_loss_sum
            loss.backward()
            batches_seen += 1

            # add max grad clipping
            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # total_loss += loss.item()

            # log information
            # if batch_idx % self.log_step == 0:
            #     self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
            #         epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        # if self.lr_decay:
        #     self.lr_scheduler.step()
        return train_epoch_loss, batches_seen

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        batches_seen = 0
        task_level = 1
        for epoch in range(1, self.epochs + 1):
            if epoch % self.step_size == 0:
                task_level += 1
            if task_level > self.args.y_offsets:
                task_level = self.args.y_offsets
            train_epoch_loss, batches_seen = self.train_epoch(epoch, batches_seen, task_level)
            val_epoch_loss = self.val_epoch(epoch)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.early_stop:
                if not_improved_count == self.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop_patience))
                    break
            if best_state:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
        np.save(self.log_dir + '/{}_train_loss.npy'.format(self.dataset), numpy.array(train_loss_list))
        np.save(self.log_dir + '/{}_val_loss.npy'.format(self.dataset), numpy.array(val_loss_list))
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        torch.save(best_model, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)
        self.model.load_state_dict(best_model)
        self.test(self.model, self.test_loader, self.scaler, self.logger, None, self.device,
                  self.log_dir, self.dataset, self.mae_thresh, self.mape_thresh)

    @staticmethod
    def test(model, data_loader, scaler, logger, path, device, log_dir, dataset, mae_thresh, mape_thresh):
        if path is not None:
            state_dict = torch.load(path, map_location=device)
            # state_dict = check_point['state_dict']
            model.load_state_dict(state_dict)
            # model.to(device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                output, _ = model(x)
                output = scaler.inverse_transform(output).cpu().numpy()
                y = scaler.inverse_transform(y).cpu().numpy()
                y_true.append(y)
                y_pred.append(output)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        np.save(log_dir + '/{}_true.npy'.format(dataset), y_true)
        np.save(log_dir + '/{}_pred.npy'.format(dataset), y_pred)
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                          mae_thresh, mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, mae_thresh, mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))


class NNITrainer(Trainer):
    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        batches_seen = 0
        task_level = 1
        for epoch in range(1, self.epochs + 1):
            if epoch % self.step_size == 0:
                task_level += 1
            if task_level > self.args.y_offsets:
                task_level = self.args.y_offsets
            train_epoch_loss, batches_seen = self.train_epoch(epoch, batches_seen, task_level)
            val_epoch_loss = self.val_epoch(epoch)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            nni.report_intermediate_result({"val loss": val_epoch_loss, "train loss": train_epoch_loss})
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.early_stop:
                if not_improved_count == self.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop_patience))
                    break
            if best_state:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
        np.save(self.log_dir + '/{}_train_loss.npy'.format(self.dataset), numpy.array(train_loss_list))
        np.save(self.log_dir + '/{}_val_loss.npy'.format(self.dataset), numpy.array(val_loss_list))
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        torch.save(best_model, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)
        self.model.load_state_dict(best_model)
        self.test(self.model, self.test_loader, self.scaler, self.logger, None, self.device,
                  self.log_dir, self.dataset, self.mae_thresh, self.mape_thresh)

    @staticmethod
    def test(model, data_loader, scaler, logger, path, device, log_dir, dataset, mae_thresh, mape_thresh):
        if path is not None:
            state_dict = torch.load(path, map_location=device)
            # state_dict = check_point['state_dict']
            model.load_state_dict(state_dict)
            # model.to(device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                output, _ = model(x)
                output = scaler.inverse_transform(output).cpu().numpy()
                y = scaler.inverse_transform(y).cpu().numpy()
                y_true.append(y)
                y_pred.append(output)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        np.save(log_dir + '/{}_true.npy'.format(dataset), y_true)
        np.save(log_dir + '/{}_pred.npy'.format(dataset), y_pred)
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                          mae_thresh, mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, mae_thresh, mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))
        nni.report_final_result({"default": float(mae), "MAE": float(mae), "MAPE": float(mape), "RMSE": float(rmse)})


def plot(logger, dataset):
    train_loss = np.load(os.path.join(logger, dataset + "_train_loss.npy"))
    val_loss = np.load(os.path.join(logger, dataset + "_train_loss.npy"))
    x = list(range(len(train_loss)))
    plt.figure(1)
    plt.plot(x, train_loss, color='b', linestyle='-', label="train loss")
    plt.plot(x, val_loss, color='r', linestyle='-', label='val loss')
    plt.xlabel('x')
    plt.ylabel('loss')
    plt.legend(loc="best")
    plt.show()


class DDPTrainer(object):
    def __init__(self, model, model_name, loss, optimizer, train_loader, val_loader,
                 test_loader, scaler, lr_scheduler, device, log_dir, grad_norm,
                 max_grad_norm, log_step, lr_decay, epochs, early_stop, early_stop_patience, mae_thresh, mape_thresh,
                 dataset, decay_interval, decay_r, init_r, final_r, l, use_curriculum_learning=True, step_size=5,
                 args=None, num_node=None):
        super(DDPTrainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader)
        self.log_dir = log_dir
        self.best_path = os.path.join(log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(log_dir, 'loss.png')
        # log
        if dist.get_rank() == 0:
            if os.path.isdir(log_dir) is False:
                os.makedirs(log_dir, exist_ok=True)
            self.logger = get_logger(log_dir, name=model_name, debug=False)
            self.logger.info('Experiment log path in: {}'.format(log_dir))
            if args is not None:
                self.logger.info(args)
            total_param = print_model_parameters(model, only_num=False)
            self.logger.info(self.model)
            self.logger.info("Total params: {}".format(str(total_param)))
        self.device = device
        self.grad_norm = grad_norm
        self.max_grad_norm = max_grad_norm
        self.log_step = log_step
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.mae_thresh = mae_thresh
        self.mape_thresh = mape_thresh
        self.dataset = dataset
        self.decay_interval = decay_interval
        self.decay_r = decay_r
        self.init_r = init_r
        self.final_r = final_r
        self.l = l
        self.use_curriculum_learning = use_curriculum_learning
        self.step_size = step_size
        self.args = args
        self.num_node = num_node

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model(x)
                loss = self.loss(output, y)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(self.val_loader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch, batches_seen, task_level=None):
        self.model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            x = x.to(self.device)
            y = y.to(self.device)
            output, structure_kl_loss_sum = self.model(x, decay_interval=self.decay_interval, decay_r=self.decay_r,
                                                       current_epoch=epoch, init_r=self.init_r, final_r=self.final_r)
            if self.use_curriculum_learning and task_level is not None:
                loss = self.loss(output[:, :task_level, ...], y[:, :task_level, ...])
            else:
                loss = self.loss(output, y)
            if dist.get_rank() == 0 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
            total_loss += loss.item()
            loss = loss + self.l * structure_kl_loss_sum
            loss.backward()
            batches_seen += 1

            # add max grad clipping
            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # total_loss += loss.item()

            # log information
            # if batch_idx % self.log_step == 0:
            #     self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
            #         epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        if dist.get_rank() == 0:
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        # if self.lr_decay:
        #     self.lr_scheduler.step()
        return train_epoch_loss, batches_seen

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        batches_seen = 0
        task_level = 1
        for epoch in range(1, self.epochs + 1):
            if epoch % self.step_size == 0:
                task_level += 1
            if task_level > self.args.y_offsets:
                task_level = self.args.y_offsets
            self.train_loader.sampler.set_epoch(epoch)
            train_epoch_loss, batches_seen = self.train_epoch(epoch, batches_seen, task_level)
            val_epoch_loss = self.val_epoch(epoch)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if dist.get_rank() == 0:
                if train_epoch_loss > 1e6:
                    self.logger.warning('Gradient explosion detected. Ending...')
                    break
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False
                if self.early_stop:
                    if not_improved_count == self.early_stop_patience:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.early_stop_patience))
                        break
                if best_state:
                    self.logger.info('*********************************Current best model saved!')
                    best_model = copy.deepcopy(self.model.moudle.state_dict())
        if dist.get_rank() == 0:
            np.save(self.log_dir + '/{}_train_loss.npy'.format(self.dataset), numpy.array(train_loss_list))
            np.save(self.log_dir + '/{}_val_loss.npy'.format(self.dataset), numpy.array(val_loss_list))
            training_time = time.time() - start_time
            self.logger.info(
                "Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            # self.model.load_state_dict(best_model)
            model = LDSTGNN(1, self.args.nb_block, self.args.input_dim, self.args.K, self.args.nb_filter, 1,
                            self.args.y_offsets, self.args.x_offsets, self.num_node, self.args.d_model, self.args.d_k,
                            self.args.d_v, self.args.n_heads, self.args.dropout).to(self.device)
            model.load_state_dict(best_model)
            self.test(model, self.test_loader, self.scaler, self.logger, None, self.device,
                      self.log_dir, self.dataset, self.mae_thresh, self.mape_thresh)

    @staticmethod
    def test(model, data_loader, scaler, logger, path, device, log_dir, dataset, mae_thresh, mape_thresh):
        if path is not None:
            state_dict = torch.load(path, map_location=device)
            # state_dict = check_point['state_dict']
            model.load_state_dict(state_dict)
            # model.to(device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                output, _ = model(x)
                output = scaler.inverse_transform(output).cpu().numpy()
                y = scaler.inverse_transform(y).cpu().numpy()
                y_true.append(y)
                y_pred.append(output)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        np.save(log_dir + '/{}_true.npy'.format(dataset), y_true)
        np.save(log_dir + '/{}_pred.npy'.format(dataset), y_pred)
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                          mae_thresh, mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, mae_thresh, mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))
