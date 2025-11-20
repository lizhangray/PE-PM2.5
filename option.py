import os, argparse
import torch, warnings
import time


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()


# 数据集相关 -----------------------------------------------------------------------------------------
parser.add_argument('--data_root', type=str, default=r'D:\workplace\project\PM2.5数据集\Heshan_imgset')
parser.add_argument('--patch_size', type=int, default=256, help='resized patch size for training')
parser.add_argument('--DS_type', type=str, default='DIS', help='DC, DS, DIS, DCAP')
parser.add_argument('--DC_patch', type=int, default=15, help='Patch size of Dark channel')


# 网络训练、推断相关 -----------------------------------------------------------------------------------
parser.add_argument('--run_type', type=str, default='train', help='train, eval, resume')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')

parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')

parser.add_argument('--net', type=str, default='PM_Pie_Net', help='the network to be use')
parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone of PM_Single_Net, swint')
parser.add_argument('--freeze_fea', action='store_true', help='freeze features, simply train the last FC')

parser.add_argument('--model_dir', type=str, default='./trained_models/model.pk')

parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use vgg pretrained parameters')
parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
parser.set_defaults(pretrained=True)

# 审稿人要求对比试验
parser.add_argument('--LReLU', dest='LReLU', action='store_true', help='ReLU for Auxilliary branch')
parser.add_argument('--ReLU', dest='LReLU', action='store_false')
parser.set_defaults(LReLU=True)

# 损失函数相关 --------------------------------------------------------------------------------------
parser.add_argument('--loss', type=str, default='L2', help='L1, L2, Focal-R, BMC')
parser.add_argument('--bin_width', type=int, default=10, help='histogram bin width')
parser.add_argument('--lds_clip', type=int, default=90, help='percentile: 70,80,90,100')
parser.add_argument('--lds_ks', type=int, default=5, help='Gaussian window size')

parser.add_argument('--balance', action='store_true', help='follow re_weight args')
parser.add_argument('--balance_type', type=str, default=r'LDS_bin', help='LDS, resample, CBL, LDS_bin')


# 接收命令行参数 -----------------------------------------------------------------
cur_time_str = time.strftime("%m%d-%H%M", time.localtime())

opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = opt.net + '_' + opt.backbone

if 'eval' == opt.run_type:
	exp_name = r'./eval_' + cur_time_str
else:
	exp_name = r'./exp_' + cur_time_str

dir_trained_models = os.path.join(exp_name, r"trained_models")
dir_numpy_files = os.path.join(exp_name, r"numpy_files")
dir_logs = os.path.join(exp_name, r"logs")


def make_dirs():
	if not os.path.exists(exp_name):
		os.mkdir(exp_name)
	if not os.path.exists(dir_trained_models):
		os.mkdir(dir_trained_models)
	if not os.path.exists(dir_numpy_files):
		os.mkdir(dir_numpy_files)
	if not os.path.exists(dir_logs):
		os.mkdir(dir_logs)