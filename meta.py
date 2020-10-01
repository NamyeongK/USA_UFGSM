import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy

from collections import defaultdict
import math

def list_average(arr):
    arr_tmp = arr[0].reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].reshape(-1)))
    arr_avg = np.average(arr_tmp)
    del arr_tmp
    return arr_avg

def tensor_sum(arr):
    arr_tmp = arr[0].cpu().reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].cpu().reshape(-1)))
    arr_avg = np.sum(arr_tmp)
    del arr_tmp
    return arr_avg

def tensor_abs_sum(arr):
    arr_tmp = arr[0].cpu().reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].cpu().reshape(-1)))
    arr_avg = np.sum(np.abs(arr_tmp))
    del arr_tmp
    return arr_avg

def list_sum(arr):
    arr_tmp = arr[0].reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].reshape(-1)))
    arr_avg = np.sum(arr_tmp)
    del arr_tmp
    return arr_avg

def list_abs_sum(arr):
    arr_tmp = arr[0].cpu().reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].reshape(-1)))
    arr_avg = np.sum(np.abs(arr_tmp))
    del arr_tmp
    return arr_avg

def list_flat(arr):
    arr_tmp = arr[0].reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].reshape(-1)))
    return arr_tmp

def list_max(arr):
    arr_tmp = arr[0].reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].reshape(-1)))
    arr_max = np.max(arr_tmp)
    del arr_tmp
    return arr_max

def list_min(arr):
    arr_tmp = arr[0].reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].reshape(-1)))
    arr_min = np.min(arr_tmp)
    del arr_tmp
    return arr_min

def list_find_idx_value(arr, idx=0):
    arr_tmp = arr[0].reshape(-1)
    for loop in range(1, len(arr)):
        arr_tmp = np.concatenate((arr_tmp, arr[loop].reshape(-1)))
    return_value = arr_tmp[idx]
    del arr_tmp
    return return_value


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.optim = args.optim
        self.args = args

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry, mode = 'train', modelfile = 'save/model.pth', pertub_scale=0.05, num_ensemble=5, fgsm_epsilon=0.05):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        def fgsm_attack(input_data, epsilon, data_grad):
            sign_data_grad = data_grad[0].sign()
            perturbed_image = input_data + epsilon * sign_data_grad
            return perturbed_image

        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        net = deepcopy(self.net) # for init params

        if mode == 'test':
            tmp = torch.load(modelfile)
            net.vars = deepcopy(tmp.net.vars)
            net.vars_bn = deepcopy(tmp.net.vars_bn)

        if self.optim == 'adam':
            meta_optim = optim.Adam(net.parameters(), lr=self.update_lr)
        elif self.optim == 'sgd':
            meta_optim = optim.SGD(net.parameters(), lr=self.update_lr)
        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        if self.optim == 'org':
            grad = torch.autograd.grad(loss, net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        if self.optim != 'org':
            meta_optim.zero_grad()
            loss.backward()
            meta_optim.step()

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            if self.optim != 'org':
                logits_q = net(x_qry, net.parameters(), bn_training=True)
            else:
                logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            if self.optim != 'org':
                logits = net(x_spt, net.parameters(), bn_training=True)
            else:
                logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)

            if self.optim != 'org':
                meta_optim.zero_grad()
                loss.backward()
                meta_optim.step()
                logits_q = net(x_qry, net.parameters(), bn_training=True)
            else:
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = net(x_qry, fast_weights, bn_training=True)

            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

                pred_s = F.softmax(logits, dim=1).argmax(dim=1)
                correct_s = torch.eq(pred_s, y_spt).sum().item()  # convert to numpy

        accs = np.array(corrects) / querysz
        if mode != 'test':
            return accs
        del(net)
        
        ## Baseline + Adversarial training ##
        if self.args.ad_train_org:
            corrects = [0 for _ in range(self.update_step_test + 1)]
            net = deepcopy(self.net)
            net.vars = deepcopy(tmp.net.vars)
            net.vars_bn = deepcopy(tmp.net.vars_bn)

            if self.optim == 'adam':
                meta_optim = optim.Adam(net.parameters(), lr=self.update_lr)
            elif self.optim == 'sgd':
                meta_optim = optim.SGD(net.parameters(), lr=self.update_lr)
            x_spt.requires_grad = True
            
            logits = net(x_spt)
            loss = F.cross_entropy(logits, y_spt)
            if self.optim == 'org':
                grad = torch.autograd.grad(loss, net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

            with torch.no_grad():
                logits_q = net(x_qry, net.parameters(), bn_training=True)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[0] = corrects[0] + correct

            if self.optim != 'org':
                grad_input = torch.autograd.grad(loss, x_spt, retain_graph=True)
                pertub_x_spt = fgsm_attack(deepcopy(x_spt), epsilon=fgsm_epsilon, data_grad=grad_input)

                logits_ad = net(pertub_x_spt, bn_training=True)
                loss_ad = F.cross_entropy(logits_ad, y_spt)
                loss = loss + loss_ad
                meta_optim.zero_grad()
                loss.backward()
                meta_optim.step()

            with torch.no_grad():
                if self.optim != 'org':
                    logits_q = net(x_qry, net.parameters(), bn_training=True)
                else:
                    logits_q = net(x_qry, fast_weights, bn_training=True)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step_test):
                if self.optim != 'org':
                    logits = net(x_spt, net.parameters(), bn_training=True)
                else:
                    logits = net(x_spt, fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt)

                if self.optim != 'org':
                    grad_input = torch.autograd.grad(loss, x_spt, retain_graph=True)
                    pertub_x_spt = fgsm_attack(deepcopy(x_spt), epsilon=fgsm_epsilon, data_grad=grad_input)
                    logits_ad = net(pertub_x_spt, bn_training=True)
                    loss_ad = F.cross_entropy(logits_ad, y_spt)

                    loss = loss + loss_ad
                    meta_optim.zero_grad()
                    loss.backward()
                    meta_optim.step()
                    logits_q = net(x_qry, net.parameters(), bn_training=True)
                else:
                    grad = torch.autograd.grad(loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                    logits_q = net(x_qry, fast_weights, bn_training=True)

                loss_q = F.cross_entropy(logits_q, y_qry)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

                    pred_s = F.softmax(logits, dim=1).argmax(dim=1)
                    correct_s = torch.eq(pred_s, y_spt).sum().item()  # convert to numpy

            accs_ad = np.array(corrects) / querysz
            del(net)
        


        def fn_USO(ensemble_uncertainty):
            meta_fast_lr = list()
            layer_uncert = list()
            for idx_layer in range(len(ensemble_uncertainty)):
                kernel_sum = 0.0
                kernel_num = ensemble_uncertainty[idx_layer].shape[0]
                for idx_kernel in range(kernel_num):
                    kernel_sum += np.mean(ensemble_uncertainty[idx_layer][idx_kernel])
                layer_uncert.append(kernel_sum / kernel_num)
            max_std = max(layer_uncert)
            min_std = min(layer_uncert)
            for idx_layer in range(len(layer_uncert)):
                tmp_var = layer_uncert[idx_layer]
                tmp_var = max_std - tmp_var + min_std
                tmp_arr = torch.ones(ensemble_uncertainty[idx_layer].shape) * tmp_var
                meta_fast_lr.append(tmp_arr)
            lr_avg = list_average(meta_fast_lr)
            for idx_layer in range(len(meta_fast_lr)):
                meta_fast_lr[idx_layer] = meta_fast_lr[idx_layer]*self.update_lr/lr_avg
            return meta_fast_lr

        def fn_train_ensemble(parent, x_spt, y_spt, x_qry, y_qry):
            net_list = list()
            update_step = self.update_step_test
            state_list = list()
            
            for loop in range(num_ensemble + 1):
                state_list.append(defaultdict(dict))
                net_list.append(deepcopy(self.net))
                for layer in range(len(net_list[loop].parameters())):
                    net_list[loop].vars[layer].data = deepcopy(parent[layer].data)

            list_fast_weights = list()

            for loop in range(num_ensemble):                
                tmp_layer = list()
                for layer in range(len(net_list[loop].parameters())):
                    perturb = torch.from_numpy(np.float32(np.random.normal(0, pertub_scale, net_list[loop].parameters()[layer].shape)) + 1).to('cuda')
                    tmp_layer.append(net_list[loop].parameters()[layer] * perturb)
                list_fast_weights.append(tmp_layer)
                
            noise_x_spt = deepcopy(x_spt_org)                    
            aug_x_spt = deepcopy(x_spt)
            aug_y_spt = deepcopy(y_spt)

            for step in range(update_step):
                en_input_grad = list()
                for loop in range(num_ensemble):
                    noise_x_spt.requires_grad = True
                    logits = net_list[loop](noise_x_spt, list_fast_weights[loop], bn_training=True)
                    loss = F.cross_entropy(logits, y_spt)

                    grad_input = torch.autograd.grad(loss, noise_x_spt, retain_graph=True)
                    en_input_grad.append(np.array(grad_input[0].detach().cpu()))
                    pertub_x_spt = fgsm_attack(deepcopy(noise_x_spt), epsilon=fgsm_epsilon, data_grad=grad_input)
                    
                    if self.args.enaug is True:
                        aug_x_spt = torch.cat((aug_x_spt, pertub_x_spt))
                        aug_y_spt = torch.cat((aug_y_spt, y_spt))

                    net_list[loop].zero_grad()
                    logits_ad = net_list[loop](pertub_x_spt, list_fast_weights[loop], bn_training=True)
                    loss_ad = F.cross_entropy(logits_ad, y_spt)
                    total_loss = loss + loss_ad
                    grad = torch.autograd.grad(total_loss, list_fast_weights[loop])

                    if self.args.org_lr == False and step > 0:
                       if self.args.adaptive is False:
                           list_fast_weights[loop] = list(map(lambda p: p[1] - (p[2] * p[0]), zip(grad, list_fast_weights[loop], uso_ss)))
                       else:
                           list_fast_weights[loop] = my_adam(list_fast_weights[loop], grad, state_list[loop], uso_ss)
                    else:
                       if self.args.adaptive is False:
                           list_fast_weights[loop] = list(map(lambda p: p[1] - (self.update_lr * p[0]), zip(grad, list_fast_weights[loop])))
                       else:
                            list_fast_weights[loop] = my_adam(list_fast_weights[loop], grad, state_list[loop], self.update_lr)
                        
                ensemble_uncertainty = list()
                with torch.no_grad():
                    for layer in range(len(net.parameters())):
                        layer_weights = list()
                        for ensemble in range(num_ensemble):
                            layer_weights.append(list_fast_weights[ensemble][layer].detach().cpu().numpy())
                        ensemble_uncertainty.append(np.array(layer_weights).std(axis=0))
                uso_ss = fn_USO(ensemble_uncertainty)
                uso_ss = list(map(lambda p: torch.from_numpy(np.asarray(p, dtype=np.float32)).cuda(), uso_ss))

                with torch.no_grad():
                    input_grad_uncertainty = torch.from_numpy(np.array(en_input_grad).std(axis=0)).cuda()

                # UFGSM
                x_spt_org.requires_grad = True
                logits = net_list[num_ensemble](x_spt_org)
                loss = F.cross_entropy(logits, y_spt)

                grad_input = torch.autograd.grad(loss, x_spt_org, retain_graph=True)
                sign_data_grad = grad_input[0].sign()

                min_max_norm_per_image_igu = torch.clone(input_grad_uncertainty)
                for loop in range(len(x_spt_org)):
                    tmp_min = torch.min(input_grad_uncertainty[loop])
                    tmp_max = torch.max(input_grad_uncertainty[loop])
                    min_max_norm_per_image_igu[loop] = (min_max_norm_per_image_igu[loop] - tmp_min) / (tmp_max - tmp_min)

                aug_image = x_spt_org + min_max_norm_per_image_igu * fgsm_epsilon * sign_data_grad
                aug_x_spt = torch.cat((aug_x_spt, aug_image))
                aug_y_spt = torch.cat((aug_y_spt, y_spt))

            del net_list
            return ensemble_uncertainty, aug_x_spt[len(x_spt_org):], aug_y_spt[len(y_spt_org):]
        
        def my_adam(src, grad, hist_state, lr):
            dst = list()
            for idx, p in enumerate(src):
                #print(idx, p.shape)
                state = hist_state[idx]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1 = 0.9
                beta2 = 0.999
                eps = 1e-8

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad[idx])   #m_hat
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad[idx], grad[idx])  #v_hat

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                if type(lr) is list:
                    step_size = lr[idx] / bias_correction1
                else:
                    step_size = lr / bias_correction1
                p.data = p.data -step_size*(exp_avg/denom)

                dst.append(p)
            return dst
        state = defaultdict(dict)
        net = deepcopy(self.net)
        net.vars = deepcopy(tmp.net.vars)
        net.vars_bn = deepcopy(tmp.net.vars_bn)
        x_spt_org = deepcopy(x_spt)
        y_spt_org = deepcopy(y_spt)
        corrects_new = [0 for _ in range(self.update_step_test + 1)]

        ensemble_uncertainty, aug_x_spt, aug_y_spt = fn_train_ensemble(net.vars, x_spt_org, y_spt_org, x_qry, y_qry)
        meta_fast_lr = fn_USO(ensemble_uncertainty)
        meta_fast_lr = list(map(lambda p: torch.from_numpy(np.asarray(p, dtype=np.float32)).cuda(), meta_fast_lr))
         
        x_spt_aug = aug_x_spt
        y_spt_aug = aug_y_spt

        x_spt.requires_grad = True
        net.zero_grad()
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        
        grad_input = torch.autograd.grad(loss, x_spt, retain_graph=True)
        pertub_x_spt = fgsm_attack(deepcopy(x_spt), epsilon=fgsm_epsilon, data_grad=grad_input)
        logits_ad = net(pertub_x_spt)
        loss_ad = F.cross_entropy(logits_ad, y_spt)

        logits_ag = net(x_spt_aug[len(x_spt_org):])
        loss_ag = F.cross_entropy(logits_ag, y_spt_aug[len(y_spt_org):])
        
        loss = loss + loss_ad + loss_ag

        grad = torch.autograd.grad(loss, net.parameters())

        if self.args.adaptive is False:
            fast_weights = list(map(lambda p: p[1] - (p[2] * p[0]), zip(grad, net.parameters(), meta_fast_lr)))
        else:
            fast_weights = my_adam(deepcopy(list(net.parameters())), grad, state, meta_fast_lr)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]                        
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects_new[0] = corrects_new[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects_new[1] = corrects_new[1] + correct

        for k in range(1, self.update_step_test):
            net.zero_grad()
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            grad_input = torch.autograd.grad(loss, x_spt, retain_graph=True)
            pertub_x_spt = fgsm_attack(deepcopy(x_spt), epsilon=fgsm_epsilon, data_grad=grad_input)

            logits_ad = net(pertub_x_spt, fast_weights, bn_training=True)
            loss_ad = F.cross_entropy(logits_ad, y_spt)

            logits_ag = net(x_spt_aug[len(x_spt_org):], fast_weights, bn_training=True)
            loss_ag = F.cross_entropy(logits_ag, y_spt_aug[len(y_spt_org):])
            loss = loss + loss_ad + loss_ag
            grad = torch.autograd.grad(loss, fast_weights)

            if self.args.adaptive is False:
                fast_weights = list(map(lambda p: p[1] - (p[2] * p[0]), zip(grad, fast_weights, meta_fast_lr)))
            else:
                fast_weights = my_adam(deepcopy(fast_weights), grad, state, meta_fast_lr)

            with torch.no_grad():
                logits_q = net(x_qry, fast_weights, bn_training=True)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects_new[k + 1] = corrects_new[k + 1] + correct
                
        del net

        accs_meta = np.array(corrects_new) / querysz
        
        if self.args.ad_train_org is True:
            return (accs, accs_ad, accs_meta)
        else:
            return (accs, accs_meta)

def main():
    pass


if __name__ == '__main__':
    main()

