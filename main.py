import os
import yaml
import argparse

import src.utils as U
#from src.processor import Processor
from src.processor import Processor
#from src.visualizer import Visualizer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = Init_parameters()

    # Update parameters by yaml
    args = parser.parse_args()
    if os.path.exists('./configs/' + args.config_id + '.yaml'):
        with open('./configs/' + args.config_id + '.yaml', 'r') as f:
            yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist the parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this config: {}'.format(args.config_id))

    # Update parameters by cmd
    args = parser.parse_args()

    # Show parameters
    # print('\n************************************************')
    # print('The running config is presented as follows:')
    # v = vars(args)
    # for i in v.keys():
    #     print('{}: {}'.format(i, v[i]))
    # print('************************************************\n')

    # Processing
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, args.gpus)))
    if args.visualization:
        #if args.extract:
            p = Processor(args)
            #print('Start visualize.....')
            p.start()

        # print('Starting visualizing ...')
        # v = Visualizer(args)
        # v.show_wrong_sample()
        # v.show_important_joints()
        # v.show_heatmap()
        # v.show_skeleton()
        # print('Finish visualizing!')

    else:
        p = Processor(args)
        p.logger.info('\n************************************************')
        p.logger.info('The running config is presented as follows:')
        v = vars(args)
        for i in v.keys():
            p.logger.info('{}: {}'.format(i, v[i]))
        p.logger.info('************************************************\n')

        p.start()
        if args.extract:
            p.extract()


def Init_parameters():
    parser = argparse.ArgumentParser(description='ST-GCN for First-Person Hand Action')
    # Logging
    parser.add_argument('--log_dir', '-ld', type=str, default='log', help='Log directory')
    parser.add_argument('--log_name', '-ln', type=str, default='train_logs.txt', help='Name of log file')
    # Data Path
    parser.add_argument('--path', '-p', type=str, default='/media/data3/duc/F-PHAB/FPHAB/fphab_data/', help='path to skeleton folder')
    parser.add_argument('--sampleName', '-sn', type=str, default='/media/data3/duc/F-PHAB/FPHAB/fphab_data/', help='path to sample test')

    # Config
    parser.add_argument('--config_id', '-c', type=str, default='', help='ID of the using config')
    # parser.add_argument('--test', '-t', default=False, action='store_true', help='test with subset')

    # Processing
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--test', '-test', default=False, action='store_true', help='Test')
    parser.add_argument('--trainFPHAB', '-tf', default=False, action='store_true', help='train on FPHAB dataset')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--visualization', '-v', default=False, action='store_true', help='Visualization')
    parser.add_argument('--test_sample', '-ts', default=False, action='store_true', help='test sample')

    # Program
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')

    # Visualization
    #parser.add_argument('--visualize_sample', '-vs', default=False, default=0, help='select sample from 0 ~ batch_size-1')
    # parser.add_argument('--visualize_class', '-vc', type=int, default=0, help='select class from 0 ~ 60, 0 means actural class')
    # parser.add_argument('--visualize_stream', '-vb', type=int, default=0, help='select stream from 0 ~ model_stream-1')
    # parser.add_argument('--visualize_frames', '-vf', type=int, default=[], nargs='+',
    #                     help='show specific frames from 0 ~ max_frame-1')

    # Dataloader
    parser.add_argument('--subset', '-ss', type=str, default='31', choices=['cs', '31','13','11'], help='benchmark of FPHAB dataset')
    parser.add_argument('--max_frame', '-mf', type=int, default=100, help='max frame number')
    parser.add_argument('--batch_size', '-bs', type=int, default=16, help='batch size')
    parser.add_argument('--data_transform', '-dt', type=U.str2bool, default=True,
                        help='channel 0~2: original data, channel 3~5: next_frame - now_frame, channel 6~8: skeletons_all - skeleton_2')
    parser.add_argument('--occlusion_part', '-op', type=int, nargs='+', default=[], choices=[1, 2, 3, 4, 5],
                         help='1:Thumb, 2:index, 3:Middle, 4:Ring, 5:Pinky')
    parser.add_argument('--occlusion_time', '-ot', type=int, default=0,
                         help='0 to 100, number of occlusion frames in first 100 frames')
    parser.add_argument('--fphab_model_type', '-fmt', type=str, default=None, choices=[None], help='type of model cmd dataset')
    parser.add_argument('--label_fphab_path', '-lfp', type=str, default='label_fphab.txt', help='labels for fphab')
    # Model
    # parser.add_argument('--pretrained', '-pt', type=U.str2bool, default=True, help='load pretrained baseline for each stream')
    # parser.add_argument('--model_stream', '-ms', type=int, default=3, help='number of model streams')
    parser.add_argument('--gcn_kernel_size', '-ks', type=int, nargs='+', default=[5,2], help='[temporal_window_size, spatial_max_distance]')
    parser.add_argument('--drop_prob', '-dp', type=int, default=0.5, help='dropout probability')
    parser.add_argument('--spatial_strategy', '-st', default=False, action='store_true', help='spatial configuration')
    parser.add_argument('--trainBaseline', '-tb', default=False, action='store_true', help='training baseline')

    # Optimizer
    parser.add_argument('--optimizer', '-opt', default='SGD', help='type of optimizer')
    parser.add_argument('--max_epoch', '-me', type=int, default=50, help='max training epoch')
    parser.add_argument('--learning_rate', '-lr', type=int, default=0.1, help='initial learning rate')
    parser.add_argument('--adjust_lr', '-al', type=int, nargs='+', default=[10,30], help='divide learning rate by 10')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--warm_up_epoch', default=0)

    return parser


if __name__ == '__main__':
    main()
