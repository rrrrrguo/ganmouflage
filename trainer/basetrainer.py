import os
from collections import defaultdict
import time
import logging
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from .checkpoints import CheckpointIO
import torch.distributed as dist
import glob

LOGGER = logging.getLogger(__name__)

class BaseTrainer(object):
    def __init__(self,
                 out_dir,
                 model_selection_metric, model_selection_mode,
                 print_every, visualize_every, checkpoint_every,
                 backup_every, validate_every, device=None, model_url=None):
        # Directories
        self.out_dir = out_dir
        self.vis_dir = os.path.join(out_dir, 'vis')
        self.log_dir = os.path.join(out_dir, 'log')
        self.model_url = model_url

        self.model_selection_metric = model_selection_metric
        if model_selection_mode == 'maximize':
            self.model_selection_sign = 1
        elif model_selection_mode == 'minimize':
            self.model_selection_sign = -1
        else:
            raise ValueError('model_selection_mode must be '
                             'either maximize or minimize.')

        self.print_every = print_every
        self.visualize_every = visualize_every
        self.checkpoint_every = checkpoint_every
        self.backup_every = backup_every
        self.validate_every = validate_every
        self.device = device

        # Checkpointer
        self.checkpoint_io = CheckpointIO(out_dir)

        # Create directories
        self.is_master=dist.get_rank()==0
        if self.is_master:
            all_dirs = [self.out_dir, self.vis_dir, self.log_dir]
            for directory in all_dirs:
                if not os.path.exists(directory):
                    os.makedirs(directory,exist_ok=True)

    def get_vis_data(self,loader):
        raise NotImplementedError

    def train(self, train_loader, val_loader, vis_loader,
              exit_after=None, n_epochs=None):
        """
        Main training method with epoch loop, validation and model selection

        args:
                train_loader
                val_loader (Validation)
                vis_loader (Visualsation during training)
        """

        # Load if checkpoint exist
        epoch_it, it, metric_val_best = self.init_training()
        print('Current best validation metric (%s): %.8f'
              % (self.model_selection_metric, metric_val_best))

        # for tensorboard
        summary_writer = SummaryWriter(os.path.join(self.log_dir))

        if self.visualize_every > 0 and self.is_master:
            #data_vis_train=self.get_vis_data(train_loader)
            if vis_loader is None:
                data_vis = self.get_vis_data(val_loader)
            else:
                data_vis = self.get_vis_data(vis_loader)
        
        # Main training loop
        t0 = time.time()
        while (n_epochs is None) or (epoch_it < n_epochs-1):
            epoch_it += 1
            dist.barrier()
            for batch in train_loader:
                it += 1
                losses = self.train_step(batch, epoch_it=epoch_it, it=it)
                
                if self.is_master:
                    if isinstance(losses, dict):
                        loss_str = []
                        for k, v in losses.items():
                            summary_writer.add_scalar('train/%s' % k, v, it)
                            loss_str.append('%s=%.4f' % (k, v))
                        loss_str = ' '.join(loss_str)
                    else:
                        summary_writer.add_scalar('train/loss', losses, it)
                        loss_str = ('loss=%.4f' % losses)

                # Print output
                if self.print_every > 0 and (it % self.print_every) == 0 and self.is_master:
                    print('[Epoch %02d] it=%03d, %s'
                          % (epoch_it, it, loss_str))

                # Visualize output
                if (self.visualize_every > 0 and (it % self.visualize_every) == 0):
                    if self.is_master:
                        print('Visualizing')
                        try:
                            #self.visualize(data_vis_train,it=it,phase='train')
                            self.visualize(data_vis,it=it,phase='valid')
                        except NotImplementedError:
                            LOGGER.warn('Visualizing method not implemented.')

                # Save checkpoint
                if (self.checkpoint_every > 0 and (it % self.checkpoint_every) == 0) and self.is_master:
                    print('Saving checkpoint')
                    self.checkpoint_io.save(
                        'model.pt', epoch_it=epoch_it, it=it,
                        loss_val_best=metric_val_best)

                # Backup if necessary
                if (self.backup_every > 0 and (it % self.backup_every) == 0) and self.is_master:
                    print('Backup checkpoint')
                    self.checkpoint_io.save(
                        'model_%d.pt' % it, epoch_it=epoch_it, it=it,
                        loss_val_best=metric_val_best)

                # Run validation and select if better
                #TODO: sync metrics from different process
                if self.validate_every > 0 and (it % self.validate_every) == 0:
                    try:
                        eval_dict = self.evaluate(val_loader)
                        #TODO: synchronize dict here
                        print(eval_dict)  
                    except NotImplementedError:
                        LOGGER.warn('Evaluation method not implemented.')
                        eval_dict = {}

                    for k, v in eval_dict.items():
                        summary_writer.add_scalar('val/%s' % k, v, it)

                    if self.model_selection_metric is not None:
                        metric_val = eval_dict[self.model_selection_metric]
                        print(
                            'Validation metric (%s): %.4f'
                            % (self.model_selection_metric, metric_val))

                        improvement = (
                            self.model_selection_sign
                            * (metric_val - metric_val_best)
                        )
                        if improvement > 0:
                            metric_val_best = metric_val
                            print('New best model (loss %.4f)'
                                  % metric_val_best)
                            self.checkpoint_io.save(
                                'model_best.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)

                # Exit if necessary
                if exit_after > 0 and (time.time() - t0) >= exit_after:
                    print('Time limit reached. Exiting.')
                    if self.is_master:
                        self.checkpoint_io.save(
                            'model.pt', epoch_it=epoch_it, it=it,
                            loss_val_best=metric_val_best)
                    exit(3)

        print('Maximum number of epochs reached. Exiting.')
        if self.is_master:
            self.checkpoint_io.save(
                'model.pt', epoch_it=epoch_it, it=it,
                loss_val_best=metric_val_best)

    def evaluate(self, val_loader):
        '''
        Evaluate model with validation data using eval_step

        args:
                data loader
        '''

        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def init_training(self):
        '''
        Init training by loading the latest checkpoint
        '''
        try:
            if self.model_url is not None:
                load_dict = self.checkpoint_io.load(self.model_url)
            else:
                if os.path.isfile(os.path.join(self.checkpoint_io.checkpoint_dir,'model.pt')):
                    load_dict = self.checkpoint_io.load('model.pt')
                else:
                    backups= glob.glob(os.path.join(self.checkpoint_io.checkpoint_dir,"model_*.pt"))
                    if len(backups)==0:
                        load_dict=dict()
                    else:
                        iterations=[int(x.split('_')[-1].split('.')[0]) for x in backups]
                        largest_it= max(iterations)
                        load_dict=self.checkpoint_io.load(f'model_{largest_it}.pt')
        except FileExistsError:
            load_dict = dict()
        epoch_it = load_dict.get('epoch_it', -1)
        it = load_dict.get('it', -1)
        metric_val_best = load_dict.get(
            'loss_val_best', -self.model_selection_sign * np.inf)
        return epoch_it, it, metric_val_best

    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        raise NotImplementedError
