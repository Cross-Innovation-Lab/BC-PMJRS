import torch
from torch import nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

# Try importing BC-PMJRS optimizer
try:
    from modules.bc_pmjrs import EnhanceSuppressOptimizer
except ImportError:
    print("Warning: EnhanceSuppressOptimizer not found. Using standard optimizer.")
    EnhanceSuppressOptimizer = None


def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    # Setup multi-GPU training with NVLink
    if hyp_params.use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with NVLink!")
        # Use DataParallel for GPU 0 and 1
        model = nn.DataParallel(model, device_ids=[0, 1])
        torch.cuda.set_device(0)
    elif hyp_params.use_cuda:
        model = model.cuda()

    # Setup optimizer
    if hyp_params.model == 'BCPMJRS' and hyp_params.use_adaptive_opt and EnhanceSuppressOptimizer is not None:
        base_optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
        optimizer = EnhanceSuppressOptimizer(base_optimizer, initial_lr=hyp_params.lr, reset_threshold=10)
    else:
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    
    criterion = getattr(nn, hyp_params.criterion)()
    
    if hyp_params.aligned or hyp_params.model in ['MULT', 'BCPMJRS']:
        ctc_criterion = None
        ctc_a2l_module, ctc_v2l_module = None, None
        ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        if hyp_params.use_cuda:
            ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
        ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
        ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)
    
    scheduler = ReduceLROnPlateau(optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer, 
                                  mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    ctc_a2l_module = settings['ctc_a2l_module']
    ctc_v2l_module = settings['ctc_v2l_module']
    ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    ctc_criterion = settings['ctc_criterion']
    scheduler = settings['scheduler']
    
    # Track performance for adaptive optimization
    best_valid_loss = float('inf')
    performance_history = []

    def train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        
        # For BC-PMJRS
        mi_loss_min_avg = 0
        mi_loss_max_avg = 0
        
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)
            
            model.zero_grad() if not hasattr(model, 'module') else model.module.zero_grad()
            if ctc_criterion is not None:
                ctc_a2l_module.zero_grad()
                ctc_v2l_module.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk
            
            # CTC processing (if needed)
            if ctc_criterion is not None:
                ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module
                audio, a2l_position = ctc_a2l_net(audio)
                vision, v2l_position = ctc_v2l_net(vision)
                
                l_len, a_len, v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
                l_position = torch.tensor([i+1 for i in range(l_len)]*batch_size).int().cpu()
                l_length = torch.tensor([l_len]*batch_size).int().cpu()
                a_length = torch.tensor([a_len]*batch_size).int().cpu()
                v_length = torch.tensor([v_len]*batch_size).int().cpu()
                
                ctc_a2l_loss = ctc_criterion(a2l_position.transpose(0,1).cpu(), l_position, a_length, l_length)
                ctc_v2l_loss = ctc_criterion(v2l_position.transpose(0,1).cpu(), l_position, v_length, l_length)
                ctc_loss = ctc_a2l_loss + ctc_v2l_loss
                ctc_loss = ctc_loss.cuda() if hyp_params.use_cuda else ctc_loss
            else:
                ctc_loss = 0
                
            # Model forward pass
            combined_loss = 0
            net = model
            
            if hyp_params.model == 'BCPMJRS':
                # BC-PMJRS with MI losses
                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    mi_loss_min_batch = 0
                    mi_loss_max_batch = 0
                    
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    vision_chunks = vision.chunk(batch_chunk, dim=0)
                    eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                    
                    for i in range(batch_chunk):
                        text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        
                        preds_i, hiddens_i, mi_min_i, mi_max_i = net(text_i, audio_i, vision_i, return_mi_loss=True)
                        
                        if hyp_params.dataset == 'iemocap':
                            preds_i = preds_i.view(-1, 2)
                            eval_attr_i = eval_attr_i.view(-1)
                        
                        raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                        raw_loss += raw_loss_i
                        mi_loss_min_batch += mi_min_i / batch_chunk
                        mi_loss_max_batch += mi_max_i / batch_chunk
                        
                        # Compute total loss for this chunk
                        chunk_loss = raw_loss_i + hyp_params.lambda_mi_min * mi_min_i + hyp_params.lambda_mi_max * mi_max_i
                        chunk_loss.backward()
                    
                    if ctc_loss != 0:
                        ctc_loss.backward()
                    
                    combined_loss = raw_loss + ctc_loss + hyp_params.lambda_mi_min * mi_loss_min_batch + hyp_params.lambda_mi_max * mi_loss_max_batch
                    mi_loss_min_avg += mi_loss_min_batch.item()
                    mi_loss_max_avg += mi_loss_max_batch.item()
                else:
                    preds, hiddens, mi_loss_min, mi_loss_max = net(text, audio, vision, return_mi_loss=True)
                    
                    if hyp_params.dataset == 'iemocap':
                        preds = preds.view(-1, 2)
                        eval_attr = eval_attr.view(-1)
                    
                    raw_loss = criterion(preds, eval_attr)
                    combined_loss = raw_loss + ctc_loss + hyp_params.lambda_mi_min * mi_loss_min + hyp_params.lambda_mi_max * mi_loss_max
                    combined_loss.backward()
                    
                    mi_loss_min_avg += mi_loss_min.item()
                    mi_loss_max_avg += mi_loss_max.item()
            else:
                # Original MULT training
                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    vision_chunks = vision.chunk(batch_chunk, dim=0)
                    eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                    
                    for i in range(batch_chunk):
                        text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        preds_i, hiddens_i = net(text_i, audio_i, vision_i)
                        
                        if hyp_params.dataset == 'iemocap':
                            preds_i = preds_i.view(-1, 2)
                            eval_attr_i = eval_attr_i.view(-1)
                        raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                        raw_loss += raw_loss_i
                        raw_loss_i.backward()
                    
                    if ctc_loss != 0:
                        ctc_loss.backward()
                    combined_loss = raw_loss + ctc_loss
                else:
                    preds, hiddens = net(text, audio, vision)
                    if hyp_params.dataset == 'iemocap':
                        preds = preds.view(-1, 2)
                        eval_attr = eval_attr.view(-1)
                    raw_loss = criterion(preds, eval_attr)
                    combined_loss = raw_loss + ctc_loss
                    combined_loss.backward()
            
            # Gradient clipping
            if ctc_criterion is not None:
                torch.nn.utils.clip_grad_norm_(ctc_a2l_module.parameters(), hyp_params.clip)
                torch.nn.utils.clip_grad_norm_(ctc_v2l_module.parameters(), hyp_params.clip)
                ctc_a2l_optimizer.step()
                ctc_v2l_optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            
            # Optimizer step
            if isinstance(optimizer, EnhanceSuppressOptimizer):
                optimizer.step(combined_loss)
            else:
                optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                
                if hyp_params.model == 'BCPMJRS':
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | MI-min {:5.4f} | MI-max {:5.4f}'.
                          format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss,
                                 mi_loss_min_avg / (i_batch + 1), mi_loss_max_avg / (i_batch + 1)))
                else:
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                          format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                
                if (ctc_a2l_module is not None) and (ctc_v2l_module is not None):
                    ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                    ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module
                    audio, _ = ctc_a2l_net(audio)
                    vision, _ = ctc_v2l_net(vision)
                
                net = model
                if hyp_params.model == 'BCPMJRS':
                    preds, _ = net(text, audio, vision, return_mi_loss=False)
                else:
                    preds, _ = net(text, audio, vision)
                
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                
                total_loss += criterion(preds, eval_attr).item() * batch_size
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        
        return avg_loss, results, truths

    # Training loop
    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion)
        val_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False)
        test_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)
        
        end = time.time()
        duration = end-start
        
        # Update learning rate based on validation loss for EnhanceSuppressOptimizer
        if isinstance(optimizer, EnhanceSuppressOptimizer):
            optimizer.update_lr_on_validation(val_loss, epoch)
        
        # Update scheduler for standard optimizers
        if hasattr(optimizer, 'base_optimizer'):
            scheduler.step(val_loss)
        else:
            scheduler.step(val_loss)
        
        # Update performance history for adaptive fusion
        if hyp_params.model == 'BCPMJRS' and hasattr(model, 'module'):
            performance_metric = 1.0 - (val_loss / best_valid_loss)
            performance_history.append(performance_metric)
            
            # Update adaptive fusion module with performance metric
            if hasattr(model.module, 'adaptive_fusion'):
                model.module.adaptive_fusion.train()
                _ = model.module.adaptive_fusion(
                    [torch.zeros(1, hyp_params.hidden_dim).cuda() for _ in range(3)],
                    performance_metric=performance_metric
                )
                model.module.adaptive_fusion.eval()
        
        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(
            epoch, duration, train_loss, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss
            best_valid_loss = val_loss

    # Load best model and evaluate
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')