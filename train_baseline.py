import os
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
from diffurec import DiffuRec
import pickle

from model import *
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',default='Beauty',type=str)#, required=True)
parser.add_argument('--train_dir',default='default',type=str)#, required=True)

parser.add_argument('--config_path',default='config/sasrec.json',type=str)#, required=True)
parser.add_argument('--model',default='SASRec',type=str)#, required=True)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_epochs', default=401, type=int)
parser.add_argument('--maxlen', default=50, type=int) #50

parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--device', default='cuda:0', type=str)

parser.add_argument('--num_preference', default=5, type=int)

parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--preprocessed_dir',default='data/preprocessed',type=str)


args = parser.parse_args()
folder = args.dataset + '_' + args.train_dir + '/' + args.model
if not os.path.isdir(folder):
    os.makedirs(folder)
with open(os.path.join(folder, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

# 模型名称到类的映射
MODEL_MAP = {
    'BERT4Rec':BERT4Rec,
    'SASRec': SASRec,
    'GRU4Rec': GRU4Rec,
    'DiffuRec':DiffuRec
}


if __name__ == '__main__':
    with open(os.path.join(args.preprocessed_dir, args.dataset + f'/preccesed_data_num_p={args.num_preference}.pkl'), 'rb') as data_file: 
        dataset = pickle.load(data_file)
        
    [user_train, user_valid, user_test, usernum, itemnum, k_and_v, preference] = dataset

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u]) + len(user_valid[u]) + len(user_test[u])
    print('The number of users: %d' % usernum )
    print('The number of items: %d' % itemnum )
    print('The number of actions: %d' % cc )
    print('Average actions of users: %.2f' % (cc / usernum ))
    print('Average actions of items: %.2f' % (cc / itemnum ))
    print('The sparsity of the dataset: %.2f%%' % ((1 - (cc / (itemnum * usernum))) * 100))
    
    
    sampler = WarpSampler(user_train, usernum, itemnum, k_and_v, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    with open(args.config_path, 'r') as config_file:
            config = json.load(config_file)

    model = MODEL_MAP.get(args.model)
    model = model(usernum, itemnum, args, config).to(args.device) 
    
    f = open(os.path.join(folder, 'log.txt'), 'w')
    
    for name, param in model.named_parameters():
        if 'item_emb' in name: continue # no init for item_emb
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        print('Evaluating', end='')
        t_test = inte_model_evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f, NDCG@5: %.4f, HR@5: %.4f)' % (t_test[0], t_test[1], t_test[2], t_test[3]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    scheduler = ReduceLROnPlateau(adam_optimizer,mode="max",factor=0.5,patience=args.patience,verbose=True)

    T = 0.0
    best_score = 0.0
    patience_counter = 0
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        t0 = time.time()
        model.train()
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, kv = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()

        print("loss in epoch {}: {}".format(epoch, loss.item()))
        t1 = time.time() - t0
        T += t1
        print("* Validation for epoch {}".format(epoch),end="")
        t_valid = inte_model_evaluate_valid(model, dataset, args)
        print('time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, NDCG@5: %.4f, HR@5: %.4f)' % (T, t_valid[0], t_valid[1], t_valid[2], t_valid[3]))

        scheduler.step(t_valid[0])
        if t_valid[0] < best_score:
            patience_counter += 1
        else:
            best_score = t_valid[0]
            patience_counter = 0
            
        if epoch == args.num_epochs or patience_counter >=  args.patience:
            model.eval()
            print('Evaluating', end='')
            t_test = inte_model_evaluate(model, dataset, args)
            print('epoch:%d, test (NDCG@10: %.4f, HR@10: %.4f, NDCG@5: %.4f, HR@5: %.4f)' % (epoch, t_test[0], t_test[1], t_test[2], t_test[3]))
            f.write(str(t_test) + '\n')
            f.flush()

            fname = 'epoch={}.maxlen={}.l2_emb={}.lr={}.patience={}'.format(epoch,args.maxlen,args.l2_emb,args.lr,args.patience)
            torch.save(model.state_dict(), os.path.join(folder, fname))
            break

    
    f.close()
    sampler.close()
    print("Done")