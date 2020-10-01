import random
import numpy as np
import os
from decimal import Decimal
import shutil


fixed_text_list = list()
fixed_text_list.append("#!/bin/bash")
fixed_text_list.append("#SBATCH --gres=gpu:v100:1")
fixed_text_list.append("#SBATCH --cpus-per-task=6")
fixed_text_list.append("#SBATCH --mem=32000M")
fixed_text_list.append("#SBATCH --time=0-00:20")
fixed_text_list.append("#SBATCH --output=%N-%j.out")

lr_list = [0.0001, 0.001, 0.01, 0.1]
opt_list = ['sgd']
num_ensemble = 5 
run_file = 'miniimagenet_train.py'

try:
    shutil.rmtree('run_script_all', ignore_errors=True)
    os.mkdir('run_script_all')
except:
    print("Error Directory")


data_list = ['cub', 'flower', 'mini', 'traffic']
model_list = ['model_5way_1shot.pth']
#model_list = ['model_5way_5shot.pth']
#model_list = ['model_5way_5shot_last_snapshot.pth']
#model_list = ['model_10way_1shot.pth']
seed_list = [222]
k_spt = 1  
k_qry = 15 
n_way =5
test_iter = 100
for idx, lr in enumerate(lr_list):
    update_lr = 0.0
    for loop in range(1, 11):
        update_lr = round(lr*loop, len(str(lr))) 
        print("Idx {} / Seed {} / lr {} / update_lr {}".format(idx, seed, lr, update_lr))
        for model_loop in range(len(model_list)):
            for data_loop in range(len(data_list)):
                for seed_loop in range(len(seed_list)):
                    seed = seed_list[seed_loop]
                    domain = data_list[data_loop]
                    model_file = model_list[model_loop]
                    opt_list[0] = 'sgd'
                    options = "--ad_train_org --enaug"
                    with open('run_script_all/{}_{}_{}_{}_{}_{}_{}.sh'.format(n_way, k_spt, model_file, seed, update_lr, domain, opt_list[0]), 'wt') as fp:
                         for loop, value in enumerate(fixed_text_list):
                             fp.write(value+'\n')
                         fp.write("python {} --mode=test --modelfile=save/{} --update_lr={} --optim={} --seed={} --num_ensemble={} --k_spt={} --n_way={} --domain={} --k_qry={} --test_iter={} {}\n".format(
                                     run_file, model_file, update_lr, opt_list[0], seed, num_ensemble, k_spt, n_way, domain, k_qry, test_iter, options))
                    opt_list[0] = 'adam'
                    options = "--ad_train_org --enaug --adaptive"
                    with open('run_script_all/{}_{}_{}_{}_{}_{}_{}.sh'.format(n_way, k_spt, model_file, seed, update_lr, domain, opt_list[0]), 'wt') as fp:
                         for loop, value in enumerate(fixed_text_list):
                             fp.write(value+'\n')
                         fp.write("python {} --mode=test --modelfile=save/{} --update_lr={} --optim={} --seed={} --num_ensemble={} --k_spt={} --n_way={} --domain={} --k_qry={} --test_iter={} {}\n".format(
                                     run_file, model_file, update_lr, opt_list[0], seed, num_ensemble, k_spt, n_way, domain, k_qry, test_iter, options))
        continue

prefix = 'sbatch --gres=gpu:v100:1'
with open('run_script_all/run_all.sh', 'wt') as fp:
    for dirname, dirnames, filenames in os.walk('run_script_all'):
        filenames.sort()
        for filename in filenames:
            fp.write("{} {}\n".format(prefix, filename))
            fp.write("sleep 10\n")

os.popen('cp run_script_all/* ./')
