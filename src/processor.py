import time
import torch
import numpy as np
from torch.backends import cudnn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import src.utils as U
from src.database import FPHAB, DatasetFactory
from src.dataprocessor import *
from src.graph import Graph
from src.visualize2D import visualize2D_anim
# from src.nets import RA_GCN, ST_GCN
from src.st_gcn_net import ST_GCN
#from src.ae_agcn    import AE_AGCN
# from src.mask import Mask
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from tqdm import tqdm
from src.logger import colorlogger as Logger
# from src.graph_as import Graph_AS
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools

import seaborn as sn
import pandas as pd
import scipy.io


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class Processor():
    def __init__(self, args):
        print('Starting preparing ...')
        self.args = args
        # Program setting
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
        self.device_type = U.check_gpu(args.gpus)
        self.device = torch.device(self.device_type)

        # Data Loader Setting
        num_class = 45
        data_shape = (3, args.max_frame, 21, 1)
        dataset_name = 'FPHAB'
        #transform data.
        transform = transforms.Compose([
            Data_transform(args.data_transform),
            Occlusion_part(args.occlusion_part),
            Occlusion_time(args.occlusion_time),
        ])
        #Load data
        self.train_loader = DataLoader(DatasetFactory(args.path, 'train', args.subset, data_shape, transform=transform, dataset_name=dataset_name, fphab_model_type=args.fphab_model_type).getDataset(),
                                       batch_size=args.batch_size, num_workers=8*len(args.gpus),
                                       pin_memory=True, shuffle=True, drop_last=True)
        #self.eval_loader = DataLoader(DatasetFactory(args.path, 'eval', args.subset, data_shape, transform=transform, dataset_name=dataset_name, fphab_model_type=args.fphab_model_type).getDataset(),
        #                              batch_size=args.batch_size, num_workers=8*len(args.gpus),
        #                              pin_memory=True, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(DatasetFactory(args.path, 'test', args.subset, data_shape, transform=transform, dataset_name=dataset_name, fphab_model_type=args.fphab_model_type).getDataset(),
                                      batch_size=args.batch_size, num_workers=8*len(args.gpus),
                                      pin_memory=True, shuffle=False, drop_last=False)
        #if self.args.test_sample:
        self.test_sample_loader=DataLoader(DatasetFactory(args.path, 'test_sample', args.subset, data_shape, transform=transform, dataset_name=dataset_name, fphab_model_type=args.fphab_model_type).getDataset(),
                                      batch_size=args.batch_size, num_workers=8*len(args.gpus),
                                      pin_memory=True, shuffle=False, drop_last=False)
        if args.data_transform:
                data_shape = (9, args.max_frame, 21, 1)

        self.logger = Logger(log_dir= args.log_dir, log_name=args.subset + args.log_name)

        # Graph Setting
        # if self.args.useAS:
        #     self.logger.info("Use AGCN's Adjacency Matrix")
        #     graph = Graph_AS()
        if self.args.spatial_strategy:
            self.logger.info("Graph was configed with spatial configuation")
            graph = Graph(max_hop=1, strategy='spatial', isFPHAB=args.trainFPHAB)
            # graph = Graph(max_hop=args.gcn_kernel_size[1])
        else:
            self.logger.info("Graph was configed with distance partition")
            graph = Graph(max_hop=args.gcn_kernel_size[1], isFPHAB=args.trainFPHAB)

        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False).to(self.device)
        self.logger.info("data shape: {}".format(data_shape))
        print(A)
        # Model Setting
        if self.args.trainBaseline:
            if args.trainFPHAB:
                self.model_name = str(args.config_id)+'_baseline_FPHAB' + args.subset + '.pkl'
            self.logger.info("train Baseline ST-GCN")
            self.model = ST_GCN(data_shape, num_class, A, args.drop_prob, args.gcn_kernel_size).to(self.device)

        self.model = nn.DataParallel(self.model)

        # Optimizer Setting
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001, nesterov=True)
        self.load_optimizer()

        # Loss Function Setting
        self.loss_func = nn.CrossEntropyLoss()

        # Mask Function Setting
        # if not self.args.trainBaseline:
        #     self.mask_func = Mask(args.model_stream, self.model.module)
        self.logger.info('Successful!')

    # Getting Model FCN Weights
    # def get_weights(self, y=None):
    #     W = []
    #     for i in range(self.args.model_stream):
    #         temp_W = self.model.module.stgcn_stream[i].fcn.weight
    #         if y is not None:
    #             temp_W = temp_W[y,:]
    #         W.append(temp_W.view(temp_W.shape[0], -1))
    #     return W

    def load_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.adjust_lr, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.args.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.logger.info('using warm up, epoch: {}'.format(self.args.warm_up_epoch))

    def adjust_learning_rate(self, epoch):
        if self.args.optimizer == 'SGD' or self.args.optimizer == 'Adam':
            if epoch < self.args.warm_up_epoch:
                lr = self.args.learning_rate * (epoch + 1) / self.args.warm_up_epoch
            else:
                lr = self.args.learning_rate * (
                        0.1 ** np.sum(epoch >= np.array(self.args.adjust_lr)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    # Learning Rate Adjusting
    # def adjust_lr(self, epoch):
    #     # LR decay
    #     if epoch in self.args.adjust_lr:
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] /= 10

    def score_report(self, y_true, y_pred, average='macro'):
        precision, recall, f1_score, _ = score(y_true, y_pred, average=average)
        return  precision, recall, f1_score

    def getLabelFPHABList(self , label_path):
        with open(label_path) as f:
            label_names = f.read().splitlines()
        return label_names

    def plot_confusion_matrix(self, cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          normalize=True):

        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy
        print(accuracy)
        if cmap is None:
            cmap = plt.get_cmap('Blues')
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.save('./results/{}.npy'.format(self.model_name), cm)
        # plt.figure(figsize=(12,12))
        # plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        # #plt.colorbar()

        # if target_names is not None:
        #     tick_marks = np.arange(len(target_names))
        #     plt.xticks(tick_marks, target_names, rotation=45, horizontalalignment="right")
        #     plt.yticks(tick_marks, target_names)



        # thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #     if normalize:
        #         plt.text(j, i, "{:0.2f}".format(cm[i, j]),
        #                  horizontalalignment="center",
        #                  color="white" if cm[i, j] > thresh else "black")
        #     else:
        #         plt.text(j, i, "{:,}".format(cm[i, j]),
        #                  horizontalalignment="center",
        #                  color="white" if cm[i, j] > thresh else "black")

        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.tight_layout()
        # # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        # #plt.show()
        # plt.savefig('{}.png'.format(self.model_name))

        df_cm = pd.DataFrame(cm,target_names,target_names)
        plt.figure(figsize=(60,60))

        sn.set(font_scale=2)#for label size
        ax=sn.heatmap(df_cm,cmap="YlGnBu", annot=True,square = True,fmt='.2f',cbar=False,
              linewidths=.2,linecolor="gold", annot_kws={"size": 22})# font size
        ax.set_title(title,fontsize='medium', fontweight='bold')
        ax.set_xlabel('Predicted label',fontsize=24, fontweight='bold')
        ax.set_ylabel('True label',fontsize=24, fontweight='bold')

        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.setp(ax.get_yticklabels(), rotation=45, horizontalalignment='right')

        plt.savefig('./results/{}.png'.format(self.model_name))
        plt.close()

    def saveConfusionMatrix(self, y_true, y_pred):
        labels_name = self.getLabelFPHABList(self.args.label_fphab_path)
        cm = metrics.confusion_matrix(y_true,y_pred)
        print("save confusion matrix")
        self.plot_confusion_matrix(cm, labels_name, normalize=True)
    
    def classify(self,y_tru,y_pred):
        labels_name = self.getLabelFPHABList(self.args.label_fphab_path)
        cm = metrics.confusion_matrix(y_true,y_pred)
        for i,name in enumerate(labels_name):






    def plot_accuracy(self, acc, test):
        plt.plot(acc,'go-',label='train', linewidth=1,markersize=3)
        plt.plot(test,'ro-',label='test',linewidth=1,markersize=3)
        plt.legend()
        plt.title('model ST-GCN')
        plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.grid()      
        plt.savefig('results/acc.png')
        plt.close()
    def plot_loss(self,loss):
        plt.plot(loss,'red')
        plt.title('model ST-GCN')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.savefig('results/loss.png')
        plt.close()

    # Training
    def train(self, epoch):
        acc, num_sample = 0, 0
        train_loader_bar = tqdm(range(len(self.train_loader)), desc="Epoch {}".format(epoch+1), position=0)
        train_loader_desc = tqdm(total=0, position=1, bar_format='{desc}')
        for num, (x, _, y, _) in enumerate(self.train_loader):
            # Using GPU
            x = x.to(self.device)
            y = y.to(self.device)
            if self.args.trainBaseline:
                out,_ = self.model(x)
            # Calculating Output
            else:
                out, feature = self.model(x)
            # Calculating Loss
            loss = self.loss_func(out, y)
            # Loss Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Calculating Accuracies
            pred = out.max(1, keepdim=True)[1]
            #print(pred)
            acc += pred.eq(y.view_as(pred)).sum().item()
            num_sample += x.shape[0]
            train_loader_bar.update(1)
            # Print Loss
            #self.logger.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}'.format(epoch+1, self.args.max_epoch, num+1, len(self.train_loader), loss))
            train_loader_desc.set_description('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}'
                            .format(epoch+1, self.args.max_epoch, num+1, len(self.train_loader), loss))
        return acc / num_sample * 100,loss


    # Testing
    def eval(self):
        with torch.no_grad():
            acc, num_sample = 0, 0
            eval_loader_bar = tqdm(range(len(self.eval_loader)), desc="Evaluate :", position=0)
            eval_loader_desc = tqdm(total=0, position=1, bar_format='{desc}')

            for num, (x, _, y, _) in enumerate(self.eval_loader):

                # Using GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Calculating Output
                if self.args.trainBaseline:
                    out, _ = self.model(x)
                else:
                    out, _ = self.model(x)

                # Calculating Accuracies
                pred = out.max(1, keepdim=True)[1]

                acc += pred.eq(y.view_as(pred)).sum().item()
                num_sample += x.shape[0]

                # Print Progress
                eval_loader_bar.update(1)
                # Print Loss
                #self.logger.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}'.format(epoch+1, self.args.max_epoch, num+1, len(self.train_loader), loss))
                eval_loader_desc.set_description('Batch: {}/{}'.format(num+1, len(self.eval_loader)))


        return acc / num_sample * 100

    def test(self):
        with torch.no_grad():
            acc, num_sample = 0, 0
            test_loader_bar = tqdm(range(len(self.test_loader)), desc="Test :", position=0)
            test_loader_desc = tqdm(total=0, position=1, bar_format='{desc}')
            precision, recall, f1_score = 0, 0, 0
            outputs = []
            targets = []
            for num, (x, _, y, _) in enumerate(self.test_loader):

                # Using GPU
                # start = time.time()
                #print(x)
                x = x.to(self.device)
                y = y.to(self.device)
                # Calculating Output
                out, _ = self.model(x)
            
                # Calculating Accuracies
                predBatch = out.data.cpu().numpy()
                gtBatch = y.data.cpu().numpy()
                outputs.append(predBatch)
                targets.append(gtBatch)
                pred = out.max(1, keepdim=True)[1]
                #print(pred)
                #predForCalculateMetrics = out.max(1)[1]
                acc += pred.eq(y.view_as(pred)).sum().item()
                #print("y true: {}".format(y.view_as(predForCalculateMetrics).to('cpu')))
                #print("pred : {}".format(predForCalculateMetrics))
                num_sample += x.shape[0]

                # Print Progress
                test_loader_bar.update(1)
                # Print Loss
                #self.logger.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}'.format(epoch+1, self.args.max_epoch, num+1, len(self.train_loader), loss))
                test_loader_desc.set_description('Batch: {}/{}'.format(num+1, len(self.test_loader)))
            outputs = np.concatenate(outputs, axis=0)
            predictions = np.argmax(outputs, axis=-1)
            targets = np.concatenate(targets, axis=0)
            if self.args.test:
                self.saveConfusionMatrix(targets, predictions)
                self.classify(targets, predictions)
            avg_precision, avg_recall, f1_score_ = self.score_report(targets, predictions, average='macro')
        return acc / num_sample * 100, avg_precision*100, avg_recall*100, f1_score_*100
    def test_1(self):
        for num, (x, _, y, _) in enumerate(self.test_sample_loader):
    
                # Using GPU
                # start = time.time()
                #print(x)
                x = x.to(self.device)
                y = y.to(self.device)
                # Calculating Output
                out, _ = self.model(x)
            
                # Calculating Accuracies
                predBatch = out.data.cpu().numpy()
                gtBatch = y.data.cpu().numpy()
                #outputs.append(predBatch)
                #targets.append(gtBatch)
                pred = out.max(1, keepdim=True)[1]
                
        return pred
    def visualization(self):
        if self.args.visualization and self.args.test_sample:
            dir="/media/data3/duc/F-PHAB/FPHAB/datasets/t_test_sample.txt"
        else:
            dir="/media/data3/duc/F-PHAB/FPHAB/datasets/visualize.txt"
        file_names=open(dir, 'rt').read().strip().split('\n')
        label=self.test_1()
        for i,file in enumerate(file_names):
            vs=visualize2D_anim(file,int(label[i]))
            vs.anim_skel()
            print("____")




    def start(self):
        # Training Start
        start_time = time.time()
        if self.args.trainFPHAB:
            if self.args.test:
                self.logger.info('Loading Testing model ...')
                checkpoint = U.load_checkpoint(self.device_type, self.model_name)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info('Successful!\n')

                # Start testing
                self.logger.info('Starting evaluating ...')
                self.model.module.eval()
                acc, precision, recall , f1_score = self.test()
                self.logger.info('Finish evaluating!')
                self.logger.info('Best accuracy: {:2.2f}%, Total time:{:.4f}s'.format(acc, time.time()-start_time))
                self.logger.info('Best precision: {0:.2f}%, Best recall: {1:.2f}%, Best f1_score: {2:.2f}%'.
                                                        format(precision, recall, f1_score))

                #print('\n')
            elif self.args.test_sample:
                checkpoint = U.load_checkpoint(self.device_type, self.model_name)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.model.module.eval()
                x1=self.test_1()
                #x1=int(x1)
                print('result:{}'.format( x1)  ) 
            elif self.args.visualization:
                checkpoint = U.load_checkpoint(self.device_type, self.model_name)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.model.module.eval()
                print('starting vs....')
                self.visualization()


            else:
                Accuracies_plot=[]
                Loss_plot=[]
                Test_plot=[]
                for i in range(self.args.adjust_lr[-1]):
                    Test_plot.append(0)
                start_epoch, best_acc = 0, 0
                best_f1_score, best_recall, best_precision = 0, 0, 0
                if self.args.resume:
                    self.logger.info('Loading checkpoint ...')
                    checkpoint = U.load_checkpoint(self.device_type)
                    self.model.module.load_state_dict(checkpoint['model'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    start_epoch = checkpoint['epoch']
                    best_acc = checkpoint['best']
                    self.logger.info('Successful!\n')

                # Start training
                self.logger.info('Starting training ...')
                self.model.module.train()
                for epoch in range(start_epoch, self.args.max_epoch):

                    # Adjusting learning rate
                    #self.adjust_lr(epoch)
                    self.adjust_learning_rate(epoch)

                    # Training
                    acc,loss = self.train(epoch)
                    Accuracies_plot.append(acc)
                    Loss_plot.append(loss)
                    self.logger.info('\nEpoch: {}/{}, Training accuracy: {:2.2f}%, Training time: {:.4f}s\n'.format(
                        epoch+1, self.args.max_epoch, acc, time.time()-start_time))
                    '''
                    if (epoch+1) > self.args.adjust_lr[-1]:
                        self.logger.info('Evaluating for epoch {} ...'.format(epoch+1))
                        self.model.module.eval()
                        acc = self.eval()
                        self.logger.info('\nEpoch: {}/{}, Evaluating accuracy: {:2.2f}%, Evaluating time: {:.4f}s\n'.format(
                                epoch+1, self.args.max_epoch, acc, time.time()-start_time))
                        self.model.module.train()
                    '''
                

                    # Evaluating
                    is_best = False

                    if (epoch+1) > self.args.adjust_lr[-1]:
                        self.logger.info('Testing for epoch {} ...'.format(epoch+1))
                        self.model.module.eval()
                        acc, precision, recall , f1_score = self.test()
                        self.logger.info('\nEpoch: {}/{}, Testing accuracy: {:2.2f}%, Testing time: {:.4f}s'.format(
                            epoch+1, self.args.max_epoch, acc, time.time()-start_time))
                        self.logger.info('precision: {0:.2f}%, recall: {1:.2f}%, f1_score: {2:.2f}%\n'.
                                                        format(precision, recall, f1_score))
                        Test_plot.append(acc)
                        self.model.module.train()
                        if acc > best_acc:
                            best_acc = acc
                            best_f1_score = f1_score
                            best_recall = recall
                            best_precision = precision
                            is_best = True

                    # Saving model
                    U.save_checkpoint(self.model.module.state_dict(), self.optimizer.state_dict(),
                                    epoch+1, best_acc, is_best, self.model_name)
                self.plot_accuracy( Accuracies_plot, Test_plot)
                self.plot_loss( Loss_plot)
                self.logger.info('Finish training!')
                #self.logger.info('Best accuracy: {:2.2f}%, Total time: {:.4f}s'.format(best_acc, time.time()-start_time))
                self.logger.info('Best accuracy : {0:.2f}%, Best precision: {1:.2f}%, Best recall: {2:.2f}%, Best f1_score: {3:.2f}%'.
                                                        format(best_acc, best_precision, best_recall, best_f1_score))
        #elif self.test_sample:
        #    print(test_1)
        elif self.args.evaluate:
            # Loading evaluating model
            self.logger.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.device_type, self.model_name)
            self.model.module.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info('Successful!\n')

            # Start evaluating
            self.logger.info('Starting evaluating ...')
            self.model.module.eval()
            acc = self.eval()
            self.logger.info('Finish evaluating!')
            self.logger.info('Best accuracy: {:2.2f}%, Total time:{:.4f}s'.format(acc, time.time()-start_time))
        '''
        else:
            # Resuming
            start_epoch, best_acc = 0, 0
            if self.args.resume:
                self.logger.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.device_type)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best']
                self.logger.info('Successful!\n')

            # Start training
            self.logger.info('Starting training ...')
            self.model.module.train()
            for epoch in range(start_epoch, self.args.max_epoch):

                # Adjusting learning rate
                # self.adjust_lr(epoch)
                self.adjust_learning_rate(epoch)

                # Training
                acc = self.train(epoch)
                self.logger.info('Epoch: {}/{}, Training accuracy: {:2.2f}%, Training time: {:.4f}s\n'.format(
                      epoch+1, self.args.max_epoch, acc, time.time()-start_time))

                # Evaluating
                is_best = False
                if (epoch+1) > self.args.adjust_lr[-1]:
                    self.logger.info('Evaluating for epoch {} ...'.format(epoch+1))
                    self.model.module.eval()
                    acc = self.eval()
                    self.logger.info('Epoch: {}/{}, Evaluating accuracy: {:2.2f}%, Evaluating time: {:.4f}s\n'.format(
                          epoch+1, self.args.max_epoch, acc, time.time()-start_time))
                    self.model.module.train()
                    if acc > best_acc:
                        best_acc = acc
                        is_best = True

                # Saving model
                U.save_checkpoint(self.model.module.state_dict(), self.optimizer.state_dict(),
                                  epoch+1, best_acc, is_best, self.model_name)
            self.logger.info('Finish training!')
            self.logger.info('Best accuracy: {:2.2f}%, Total time: {:.4f}s'.format(best_acc, time.time()-start_time))
        '''

    def extract(self):
        self.logger.info('Starting extracting ...')
        self.model.module.eval()

        # Loading extracting model
        self.logger.info('Loading extracting model ...')
        checkpoint = U.load_checkpoint(self.device_type, self.model_name)
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('Successful!\n')

        # Loading Data
        if self.args.test:
            x, l, y, name = iter(self.test_loader).next()
        else:
            x, l, y, name = iter(self.eval_loader).next()
        # x, l, y, name = iter(self.eval_loader).next()

        # Using GPU
        x = x.to(self.device)
        y = y.to(self.device)

        # Calculating Output
        out, feature = self.model(x)
        out = F.softmax(out, dim=1)

        # Using CPU
        out = out.detach().cpu().numpy()
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        # # Loading Weight
        # weight = []
        # W = self.get_weights()
        # for i in range(self.args.model_stream):
        #     weight.append(W[i].detach().cpu().numpy())
        #     feature[i] = feature[i].detach().cpu().numpy()

        # Saving Feature
        #np.savez('./visualize.npz', feature=feature, out=out, weight=weight, label=y, location=l.numpy(), name=name)
        self.logger.info('Finish extracting!\n')
