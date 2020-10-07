import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta

from sys import argv
import time


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    start_time = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(args)
    print(argv)
    os.makedirs(args.modelfile.split('/')[0], exist_ok=True)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    mini = MiniImagenet('./dataset/mini-imagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    if args.domain == 'mini':
        mini_test = MiniImagenet('./dataset/mini-imagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                 k_query=args.k_qry,
                                 batchsz=args.test_iter, resize=args.imgsz)
        mini_val = MiniImagenet('./dataset/mini-imagenet/', mode='val', n_way=args.n_way, k_shot=args.k_spt,
                                 k_query=args.k_qry,
                                 batchsz=args.test_iter, resize=args.imgsz)
    elif args.domain == 'cub':
        print("CUB dataset")
        mini_test = MiniImagenet('./dataset/CUB_200_2011/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                 k_query=args.k_qry,
                                 batchsz=args.test_iter, resize=args.imgsz)
    elif args.domain == 'traffic':
        print("Traffic dataset")
        mini_test = MiniImagenet('./dataset/GTSRB/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                 k_query=args.k_qry,
                                 batchsz=args.test_iter, resize=args.imgsz)
    elif args.domain == 'flower':
        print("flower dataset")
        mini_test = MiniImagenet('./dataset/102flowers/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                 k_query=args.k_qry,
                                 batchsz=args.test_iter, resize=args.imgsz)
    else:
        print("Dataset Error")
        return

    if args.mode == 'test':
        count = 0
        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=6, pin_memory=True)
        accs_all_test = []

        for x_spt, y_spt, x_qry, y_qry in db_test:
            print(count)
            count += 1
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, 'test', args.modelfile, pertub_scale=args.pertub_scale, num_ensemble=args.num_ensemble, fgsm_epsilon=args.fgsm_epsilon)
            accs_all_test.append(accs)

        # [b, update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        np.set_printoptions(linewidth=1000)
        print("Running Time:", time.time()-start_time)
        print(accs)
        return


    for epoch in range(args.epoch//10000):
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=4, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                print('epoch:', epoch, 'step:', step, '\ttraining acc:', accs)

            if step % 200 == 0:
                print("Save model", args.modelfile)
                torch.save(maml, args.modelfile)
                db_test = DataLoader(mini_val, 1, shuffle=True, num_workers=4, pin_memory=True)
                accs_all_val = []
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, 'train_test')
                    accs_all_val.append(accs)
                
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=4, pin_memory=True)
                accs_all_test = []
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)
                    
                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                accs_val = np.array(accs_all_val).mean(axis=0).astype(np.float16)

                save_modelfile = "{}_{}_{}_{:0.4f}_{:0.4f}.pth".format(args.modelfile, epoch, step, accs_val[-1], accs[-1])
                print(save_modelfile)
                torch.save(maml, save_modelfile) 
                print("Val:", accs_val)
                print("Test:", accs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=150000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--dataset', type=str, help='dataset', default='mini')
    argparser.add_argument('--mode', type=str, help='train or test', default='train')
    argparser.add_argument('--modelfile', type=str, help='pretraind model path', default='save/model.pth')
    argparser.add_argument('--optim', type=str, help='optimizer for test', default='sgd')

    argparser.add_argument('--pertub_scale', type=float, help='ensemble model pertubation scale', default=0.05)
    argparser.add_argument('--fgsm_epsilon', type=float, help='fgsm epsilon', default=0.05)    
    argparser.add_argument('--num_ensemble', type=int, help='number of ensemble models', default=5)
            
    argparser.add_argument('--ad_train_org', help='baseline model training with adversarial training', action='store_true')
    argparser.add_argument('--org_lr', help='original manner', action='store_true')
    argparser.add_argument('--adaptive', help='model training for proposed method using adam', action='store_true')

    argparser.add_argument('--enaug', help='ensemble data augmentation', action='store_true')

    argparser.add_argument('--seed', type=int, help='random seed', default=222)
    argparser.add_argument('--domain', type=str, help='dataset domain', default='mini')
    argparser.add_argument('--test_iter', type=int, help='test iterations', default=100)


    args = argparser.parse_args()

    main()
