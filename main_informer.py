import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--data', type=str, default='WTH', help='data')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='WTH.csv', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='T', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--length', type=str, default=3, help='length X pred_length')
parser.add_argument('--decay_rate', type=str, default=0.5, help='fix decay rate for learning rate')


# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--train_len', type=int, default=96, help='input sequence length of Informer encoder') #96个用于训练 #default 96
parser.add_argument('--delead_len', type=int, default=48, help='start token length of Informer decoder') #48个作为预测参考 #default 48
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length') #预测长度 #default 24

#selection of Optimizer
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer, adam or adammax')

#selection of Loss Function
parser.add_argument('--lossfunction', type=str, default='mse', help='lossfunction, L1, smoothL1(HuberL1) or mse')


parser.add_argument('--layer_attn', type=str, default=0, help='attn layer of decoder')
parser.add_argument('--en_input', type=int, default=7, help='encoder input size')
parser.add_argument('--de_input', type=int, default=7, help='decoder input size')
parser.add_argument('--out_size', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--en_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--de_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')   #factor:用于决定从96个Q里面选几个Q作为明显的
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'LVMH':{'data':'LVMH.csv','T':'Close','M':[6,6,6],'S':[1,1,1],'MS':[6,6,1]},
    'S68_4':{'data':'S68_4.csv','T':'Close','M':[4,4,4],'S':[1,1,1],'MS':[4,4,1]},
    'AMZN':{'data':'AMZN.csv','T':'Close','M':[6,6,6],'S':[1,1,1],'MS':[6,6,1]},
    'GT':{'data':'GT.csv','T':'LandAndOceanAverageTemperature','M':[8,8,8],'S':[1,1,1],'MS':[8,8,1]},
    
}

data_info = data_parser.get(args.data)

# 如果data_info存在
if data_info:
    # 获取args.data_path、args.target、args.en_input、args.de_input和args.out_size的值
    args.data_path = data_info.get('data')
    args.target = data_info.get('T')
    args.en_input, args.de_input, args.out_size = data_info.get(args.features)

#args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
# 将字符串转换为整数列表
s_layers_list = []
for s_l in args.s_layers.replace(' ', '').split(','):
    s_layers_list.append(int(s_l))
args.s_layers = s_layers_list
args.detail_freq = args.freq
args.freq = args.freq[-1:]

#print('Args in experiment:')
#print(args)

Experiments = Exp_Informer

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.train_len, args.delead_len, args.pred_len,
                args.d_model, args.n_heads, args.en_layers, args.de_layers, args.d_ff, args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, ii)

    experiments = Experiments(args) # set experiments
    #print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    experiments.train(setting)
    
    #print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    experiments.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        experiments.predict(setting, True)

    torch.cuda.empty_cache()

#plot losses
train_loss = np.load('results/' + setting + '/train_loss.npy')
vali_loss = np.load('results/' + setting + '/vali_loss.npy')

error = np.load('results/' + setting + '/metrics.npy')
epoch = range(len(train_loss))

plt.plot(epoch, train_loss, label='Train Loss')
plt.plot(epoch, vali_loss, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Losses')
plt.legend()

plt.subplots_adjust(top=0.8, bottom=0.3)
plt.show()

#inference
preds = np.load('./results/'+setting+'/pred.npy')
ground_truth = np.load('./results/'+setting+'/true.npy')

def concat_seq_len(x, length, stored_file):
    xx = np.load('./results/'+ setting + stored_file)
    store = []
    for i in range(length):
        store.append(xx[i,:,-1])
    return np.array(store).reshape(-1)

long_preds = concat_seq_len(preds,args.length, '/pred.npy')
long_gt = concat_seq_len(ground_truth,args.length, '/true.npy')

plt.figure()
plt.plot(long_preds, label='GroundTruth')
plt.plot(long_gt, label='Prediction')
plt.legend()
plt.title('Prediction on Test Set')  # Add a title to the plot
plt.xlabel(f"Time, {args.freq}")  # Add an x-axis label to the plot
plt.ylabel(f"Normalised {args.target} Prediction")  # Add a y-axis label to the plot
plt.show()

#Visualise Attention

args.layer_attn = 0

Data = Dataset_Custom
timeenc = 0 if args.embed!='timeF' else 1
flag = 'test'; shuffle_flag = False; drop_last = True; batch_size = 1

data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.train_len, args.delead_len, args.pred_len],
    features=args.features,
    timeenc=timeenc,
    target=args.target, # HULL here
    freq=args.freq # 'h': hourly, 't':minutely
)
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)

args.output_attention = True

exp_arg = Experiments(args)
path = os.path.join(args.checkpoints,setting,'checkpoint.pth')
model = exp_arg.model
model.load_state_dict(torch.load(path))

idx = 0
for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
    if i!=idx:
        continue
    batch_x = batch_x.float().to(exp_arg.device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(exp_arg.device)
    batch_y_mark = batch_y_mark.float().to(exp_arg.device)
    
    dec_inp = torch.zeros_like(batch_y[:,-args.pred_len:,:]).float()
    dec_inp = torch.cat([batch_y[:,:args.delead_len,:], dec_inp], dim=1).float().to(exp_arg.device)
    
    outputs,attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

distil = 'Distil' if args.distil else 'NoDistil'
for h in range(0,8):
    plt.figure(figsize=[10,8])
    plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, args.layer_attn, h))
    A = attn[args.layer_attn][0,h].detach().cpu().numpy()
    ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
    plt.show()
    
