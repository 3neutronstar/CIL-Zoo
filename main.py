import torch
import argparse
import time
import os
import sys
from implementor.baseline import Baseline
from implementor.eeil import EEIL
from implementor.icarl import ICARL
from model.basenet import get_model
from implementor.evaluator.evaluation import Evaluation
from utils.seed import fix_seed
from utils.params import load_params, save_params


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="python main.py mode")

    parser.add_argument(
        'mode', type=str,
    )
    parser.add_argument(
        '--seed', type=int, default=1, help='fix random seed')
    parser.add_argument(
        '--model', type=str, default='resnet32', help='choose NeuralNetwork model')
    parser.add_argument(
        '--device', type=str, default='cuda', help='choose NeuralNetwork')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='set mini-batch size')
    parser.add_argument(
        '--num_workers', type=int, default=3, help='number of process you have')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5, help='set optimizer\'s weight decay')
    parser.add_argument(
        '--lr', type=float, default=2.0, help='set learning rate')
    parser.add_argument(
        '--gamma', type=float, default=0.2, help='set lr decaying rate 1/5')
    parser.add_argument(
        '--nesterov', type=bool, default=False, help='set learning rate')
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='set momentum')
    parser.add_argument(
        '--temperature', type=float, default=2.0, help='set temperature of knowledge distillation')
    parser.add_argument(
        '--epochs', type=int, default=70, help='run epochs')
    parser.add_argument(
        '--dataset', type=str, default='cifar100', help='select dataset')
    parser.add_argument(
        '--memory_size', type=int, default=2000, help='exemplar set size')
    parser.add_argument(
        '--task_size', type=int, default=5, help='the number of task 5, 10 , 20, 25, 50, etc.')
    parser.add_argument('--gpu_ids', default='0',
                        type=str, help=' ex) 0,1,2')
    parser.add_argument('--detect_anomaly', default=False,
                        type=bool, help='Detect anomaly in PyTorch')
    parser.add_argument('--lr_steps', help='lr decaying epoch determination', default=[49,63],
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--dataset_path', help='dataset path (None: /data or .\\data\\dataset\\)', default=None)

    if parser.parse_known_args(args)[0].mode.lower() == 'train':
        parser.add_argument(
            '--train_mode', '-t', type=str, default='baseline', choices=['baseline','icarl','eeil'],
            help='Choose Train Mode')

    elif parser.parse_known_args(args)[0].mode.lower() == 'eval':
        parser.add_argument(
            '--file_name', type=str, default=None,
            help='Read file name')
        parser.add_argument('--evaluator', type=str, default='baseline',
                            choices=['baseline'])

    if 'tiny-imagenet' == parser.parse_known_args(args)[0].dataset:
        parser.add_argument('--tiny_resize', type=bool,
                            default=False, help='choose 224 size or 64')

    return parser.parse_known_args(args)[0]


def main(args):
    flags = parse_args(args)
    configs = vars(flags)
    ## gpu parallel ##
    os.environ["CUDA_VISIBLE_DEVICES"] = configs['gpu_ids']
    ##################
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda and configs['device'] == 'cuda' else "cpu")
    configs['device'] = str(device)
    ## detect anomaly ##
    if configs['detect_anomaly']:
        torch.autograd.set_detect_anomaly(True)
    ####################
    ## seed ##
    fix_seed(seed=configs['seed'])
    ##########

    ## time data ##
    time_data = time.strftime(
        '%m-%d_%H-%M-%S', time.localtime(time.time()))
    ###############

    ## data save path ##
    current_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_path, 'outputs')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, time_data)) and configs['mode'] != 'eval':
        os.mkdir(os.path.join(save_path, time_data))
    ####################

    ## save configuration ##
    parallel = configs['gpu_ids']
    if configs['mode'] != 'eval':
        configs.pop('gpu_ids')
        dataset_path=configs.pop('dataset_path')
        save_params(configs, current_path, time_data)
    else:
        if configs['file_name'] is None and 'pretrained' not in configs['model']:
            raise ValueError('Please provide file name')
        elif 'pretrained' not in configs['model']:
            file_name = configs['file_name']
            seed=configs['seed']
            configs.update(load_params(
                configs, current_path, configs['file_name']))
            configs['mode'] = 'eval'
            configs['file_name'] = file_name
            configs['device'] = device
            configs['seed']=seed
            time_data=file_name
        dataset_path=configs['dataset_path']
        
    configs['gpu_ids'] = parallel
    print("="*30)
    print(configs)
    print("="*30)
    ########################

    ## Num Classes ##
    if configs['dataset'] in ['cifar10', 'fashionmnist', 'mnist', 'stl10']:
        configs['num_classes'] = 10
    elif configs['dataset'] in ['cifar100', 'aircraft100']:
        configs['num_classes'] = 100
    elif configs['dataset'] in ['flowers102']:
        configs['num_classes'] = 102
    elif configs['dataset'] in ['food101', 'caltech101']:
        configs['num_classes'] = 101
    elif configs['dataset'] == 'caltech256':
        configs['num_classes'] = 257
    elif configs['dataset'] == 'cars196':
        configs['num_classes'] = 196
    elif configs['dataset'] == 'dogs120':
        configs['num_classes'] = 120
    elif configs['dataset'] in ['tiny-imagenet', 'cub200','imagenet200']:
        configs['num_classes'] = 200
    elif configs['dataset'] == 'inat2017':
        configs['num_classes'] = 5089
    else:  # imagenet
        configs['num_classes'] = 1000
    print("Number of classes : {}".format(configs['num_classes']))
    #################

    ## Model ##
    model = get_model(configs)
    ###########

    if configs['mode'] == 'train':
        LEARNER = {
            'baseline': Baseline,
            'icarl': ICARL,
            'eeil': EEIL,
        }
        learner = LEARNER[configs['train_mode']](
            model, time_data, save_path, device, configs)
        learner.run(dataset_path)

    elif 'eval' == configs['mode']:
        EVALUATOR = {
            'baseline': Evaluation,
        }
        evaluator = EVALUATOR[configs['evaluator']](
            model, time_data, dataset_path, save_path, device, configs)
        evaluator.evaluation()


if __name__ == '__main__':
    main(sys.argv[1:])
