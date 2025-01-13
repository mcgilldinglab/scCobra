#!/usr/bin/env python

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import numpy as np
import os
import scanpy as sc
from anndata import AnnData
from typing import Union, List

import time


from .data import load_data
from .net.utils import EarlyStopping
from .metrics import batch_entropy_mixing_score, silhouette_score
from .logger import create_logger
from .plot import embedding
from .info_nce import InfoNCE, ClusterLoss, DCL

import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from torch.distributions import Normal

from torch.distributions import Normal, kl_divergence

import torch_optimizer as optim

def kl_div(mu, var):
    return kl_divergence(Normal(mu, var.sqrt()),
                         Normal(torch.zeros_like(mu),torch.ones_like(var))).sum(dim=1).mean()

 
def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    """
    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])
        
    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()
            
    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()
            
    def _check_input_dim(self, input):
        raise NotImplementedError
            
    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device) #, requires_grad=False)
        for i in range(self.n_domain):
            indices = torch.where(y == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
#                 self.bns[i].training = False
#                 out[indices] = self.bns[i](x[indices])
#                 self.bns[i].training = True
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Encoder, self).__init__()

        self.embnet = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU()
        )

        self.mu = nn.Linear(1024, z_dim)
        
        self.sigma = nn.Linear(1024, z_dim)


    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()
        
    def forward(self, x):            
   
        z = self.embnet(x)
    
        mu = self.mu(z)
        
        sigma = torch.exp(self.sigma(z))
        
        q = self.reparameterize(mu, sigma)
    
        return q, mu, sigma

class Decoder(nn.Module):
    def __init__(self, z_dim, input_dim, domain):
        super(Decoder, self).__init__()
        
        self.embnet = nn.Sequential(
            nn.Linear(z_dim, input_dim)
        )

        self.norm = DSBatchNorm(input_dim, domain)
        
    def forward(self, x, y):            
   
        x_rec = self.embnet(x)  
    
        x_rec = torch.sigmoid(self.norm(x_rec, y))
    
        return x_rec
    

class Discrminator(nn.Module):
    def __init__(self, input_dim, domian_number=1):
        super(Discrminator, self).__init__()

        self.dis_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU()
        )

        self.disc_layer = nn.Linear(1024, 1)

        self.batch_layer = nn.Linear(1024, domian_number)


    def forward(self, x):

        x = self.dis_head(x)            
   
        disc_out = self.disc_layer(x)

        batch_out = self.batch_layer(x)

        return disc_out, batch_out
    
class Batch_Discrminator(nn.Module):
    def __init__(self,  domian_number=1):
        super(Batch_Discrminator, self).__init__()

        self.dis_head = nn.Sequential(
            nn.Linear(10, 16),
            nn.GELU(),
            nn.Linear(16, domian_number),
        )


    def forward(self, x):

        batch_label = self.dis_head(x)            

        return batch_label
    
class contrastive_head(nn.Module):
    def __init__(self,  cluster_number=1):
        super(contrastive_head, self).__init__()

        self.instance_head = nn.Sequential(
            nn.Linear(10, 10),
            nn.GELU(),
            nn.Linear(10, 10),
        )

        self.cluster_head = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.GELU(),
            nn.Linear(10, cluster_number),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        instance_out = self.instance_head(x)            

        cluster_out = self.cluster_head(x)

        return instance_out, cluster_out


class scCobra_model(nn.Module):

    def __init__(self, input_dim=2000, n_domain=1, class_num=15):

        super().__init__()

        self.z_dim=10

        self.x_dim = input_dim

        self.encoder = Encoder(input_dim, self.z_dim)
        
        self.fixed_encoder = Encoder(input_dim, self.z_dim)
        
        for param, param_fixed in zip(
            self.encoder.parameters(), self.fixed_encoder.parameters()
        ):
            param_fixed.data.copy_(param.data)  # initialize
            param_fixed.requires_grad = False  # not update by gradient

        self.decoder = Decoder(self.z_dim, input_dim, n_domain)

        self.discriminator = Discrminator(self.x_dim, n_domain)

        self.batch_discriminator = Batch_Discrminator(n_domain)

        self.contrastive_head = contrastive_head(cluster_number=class_num)

        self.domain_loss = LabelSmoothingCrossEntropy()
        
        # self.contrastive_loss = InfoNCE(temperature=0.2)
        
        self.use_cluster = False

        self.contrastive_loss = DCL(temperature=0.1)

        self.cluster_loss = ClusterLoss(class_num=class_num, temperature=1)        
        
    @torch.no_grad()
    def update_fixed_encoder(self):
        """
        Update of the fixe_encoder
        """
        
        for param, param_fixed in zip(
            self.encoder.parameters(), self.fixed_encoder.parameters()
        ):
            param_fixed.data.copy_(param.data)  # initialize

    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
    def encodeBatch(
            self, 
            dataloader, 
            device='cuda', 
            out='latent', 
            batch_id=None,
            return_idx=False, 
            eval=False
        ):
        """
        Inference
        
        Parameters
        ----------
        dataloader
            An iterable over the given dataset for inference.
        device
            'cuda' or 'cpu' for . Default: 'cuda'.
        out
            The inference layer for output. If 'latent', output latent feature z. If 'impute', output imputed gene expression matrix. Default: 'latent'. 
        batch_id
            If None, use batch 0 decoder to infer for all samples. Else, use the corresponding decoder according to the sample batch id to infer for each sample.
        return_idx
            Whether return the dataloader sample index. Default: False.
        eval
            If True, set the model to evaluation mode. If False, set the model to train mode. Default: False.
        
        Returns
        -------
        Inference layer and sample index (if return_idx=True).
        """
        self.to(device)
        if eval:
            self.eval();print('eval mode')
        else:
            self.train()
        indices = np.zeros(dataloader.dataset.shape[0])
        if out == 'latent':
            output = np.zeros((dataloader.dataset.shape[0], self.z_dim))
            for x,y,idx in dataloader:

                x = x.float().to(device)
                z = self.encoder(x)[1]
                output[idx] = z.detach().cpu().numpy()
                indices[idx] = idx
                
        elif out == 'impute':
            output = np.zeros((dataloader.dataset.shape[0], self.x_dim))

            
            if batch_id in dataloader.dataset.adata.obs['batch'].cat.categories:
                batch_id = list(dataloader.dataset.adata.obs['batch'].cat.categories).index(batch_id)
            else:
                batch_id = 0

            for x,y,idx in dataloader:
                x = x.float().to(device)
                z = self.encoder(x)[1] # z, mu, var
                output[idx] = self.decoder(z, torch.LongTensor([batch_id]*len(z))).detach().cpu().numpy()
                indices[idx] = idx

        if return_idx:
            return output, indices
        else:
            return output
    
    def generate(
            self,
            dataloader, 
            embedding,
            device='cuda',  
            batch_id=None
        ):

        self.to(device)

        self.eval() 

        output = np.zeros((dataloader.dataset.shape[0], self.x_dim))

        batch_id = batch_id

        embedding = torch.tensor(embedding).float().to(device)
        
        output = self.decoder(embedding, torch.LongTensor([batch_id]*len(embedding))).detach().cpu().numpy()

        return output
    
    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, device):
        """计算WGAN-GP梯度惩罚"""
        # 随机权重用于混合真实和伪造样本
        alpha = torch.rand(real_samples.size(0), 1, device=device)
        alpha = alpha.expand_as(real_samples)
        
        # 混合真实和伪造样本
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated = interpolated.to(device)
        interpolated.requires_grad_(True)
        
        # 评估混合样本
        d_interpolated = discriminator(interpolated)[0]
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # 计算梯度的范数和梯度惩罚
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        # 梯度惩罚系数
        lambda_gp = 10
        return lambda_gp * gradient_penalty


    def fit(
            self, 
            dataloader, 
            lr=3e-4,
            max_iteration=30000,
            device='cuda',
            early_stopping=None,
            verbose=False,
        ):

        self.to(device)

        vae_optim = optim.AdaBelief(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            betas = (0.5, 0.999),
            eps = 1e-12
            )
        
        disc_optim = optim.AdaBelief(
            self.discriminator.parameters(),
            lr=lr,
            betas = (0.5, 0.999),
            eps = 1e-12
            )
        
        batch_disc_optim = optim.AdaBelief(
            self.batch_discriminator.parameters(),
            lr=lr,
            betas = (0.5, 0.999),
            eps = 1e-12
            )

        # contrastive_head_optim = optim.AdaBelief(
        #     self.contrastive_head.parameters(),
        #     lr=lr,
        #     betas = (0.5, 0.999),
        #     eps = 1e-12
        #     )

        n_epoch = int(np.ceil(max_iteration/len(dataloader)))
        
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:       
            for epoch in tq:
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)
                for i, (x, y, idx) in tk0:

                    real_data, batch_label = x.float().to(device), y.long().to(device)
                    
                    
                    '''----------------         Discriminator Update         ----------------'''

                    disc_optim.zero_grad()

                    # 编码真实数据，生成重构数据
                    q, mu, sigma = self.encoder(real_data)
                    rec_data = self.decoder(q, batch_label)

                    # 判别器输出
                    # 对真实数据输出真假判别和批次判别
                    real_disc_out, real_batch_out = self.discriminator(real_data)
                    # 对生成数据输出真假判别和批次判别
                    rec_disc_out, rec_batch_out = self.discriminator(rec_data.detach())

                    # WGAN 损失
                    disc_loss = torch.mean(rec_disc_out) - torch.mean(real_disc_out)

                    # 计算梯度惩罚
                    gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_data, rec_data, device=device)

                    # 批次分类损失
                    batch_loss_real = F.cross_entropy(real_batch_out, batch_label)
                    batch_loss_rec = F.cross_entropy(rec_batch_out, batch_label)
                    batch_loss = (batch_loss_real + batch_loss_rec) / 2

                    # 判别器总损失
                    gan_objective = disc_loss + gradient_penalty + batch_loss

                    gan_objective.backward()
                    disc_optim.step()

                    '''----------------         Batch Discriminator Update         ----------------'''

                    batch_disc_optim.zero_grad()

                    # 编码真实数据，生成重构数据

                    q, mu, sigma = self.encoder(real_data)

                    # 对生成数据输出批次判别
                    domain_label = self.batch_discriminator(mu)

                    # 批次分类损失
                    domin_loss = self.domain_loss(domain_label, batch_label)

                    domin_loss.backward()

                    batch_disc_optim.step()


                    '''----------------         VAE Update         ----------------'''

                    vae_optim.zero_grad()
                                        
                    self.update_fixed_encoder()
                        
                    # 编码真实数据，生成重构数据
                    q, mu, sigma = self.encoder(real_data)
                    rec_data = self.decoder(q, batch_label)

                    # KL 散度
                    KLD = kl_div(mu, sigma)
                    
                    # 判别器对生成数据的真假输出
                    rec_disc_out, rec_batch_out = self.discriminator(rec_data)
                    
                    # 让判别器无法分辨真假样本（GAN 损失）
                    GAN_loss = -torch.mean(rec_disc_out)

                    # 批次分类损失
                    batch_loss = F.cross_entropy(rec_batch_out, batch_label)
                                        
                    # 对比损失：利用固定编码器生成潜在表示
                    _, real_mu, _ = self.fixed_encoder(real_data)
                    _, rec_mu, _ = self.fixed_encoder(rec_data)

                    # instance_out, cluster_out = self.contrastive_head(real_mu)
                    # instance_out_rec, cluster_out_rec = self.contrastive_head(rec_mu)

                    # contrastive_loss_ = self.contrastive_loss(instance_out, instance_out_rec) + self.contrastive_loss(cluster_out, cluster_out_rec)

                    contrastive_loss_ = self.contrastive_loss(real_mu, rec_mu) + self.contrastive_loss(rec_mu, real_mu)

                    if self.use_cluster:
                        contrastive_loss_ = self.contrastive_loss(instance_out, instance_out_rec) + self.contrastive_loss(cluster_out, cluster_out_rec) + self.cluster_loss(cluster_out, cluster_out_rec)

                    
                    # 重构损失
                    rec_loss = F.binary_cross_entropy(rec_data, real_data) * real_data.size(-1)
                    # rec_loss = binary_cross_entropy(rec_data, real_data)

                    # batch_discriminator损失
                    domain_label = self.batch_discriminator(mu)
                    domin_loss = self.domain_loss(domain_label, batch_label)

                    
                    #总损失
                    
                    vae_objective = contrastive_loss_ + 0.5 * KLD + GAN_loss + rec_loss + batch_loss

                    vae_objective.backward()
                    vae_optim.step()

                    loss = {
                        'contrastive_loss': contrastive_loss_,
                        'KLD': KLD * 0.5,
                        'GAN_loss': GAN_loss,
                        'rec_loss': rec_loss,
                        'batch_loss': batch_loss,
                        'domin_loss': domin_loss,
                    }
     
                    for k,v in loss.items():
                        epoch_loss[k] += loss[k].item()
                        
                                        
                    info = ','.join(['{}={:.3f}'.format(k, v) for k in loss.items()])
                    tk0.set_postfix_str(info)
                 
                epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
                tq.set_postfix_str(epoch_info)

                early_stopping(sum(epoch_loss.values()), self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} epoch'.format(epoch+1))
                    break
                

def scCobra(
        data_list, #Union[str, AnnData, List]=None, 
        batch_categories:list=None,
        profile:str='RNA',
        batch_name:str='batch',
        min_features:int=600, 
        min_cells:int=3, 
        target_sum:int=None,
        n_top_features:int=None,
        join:str='inner', 
        batch_key:str='batch',  
        processed:bool=False,
        fraction:float=None,
        n_obs:int=None,
        use_layer:str='X',
        keep_mt:bool=False,
        backed:bool=False,
        batch_size:int=64, 
        lr:float=2e-4, 
        max_iteration:int=30000,
        seed:int=124, 
        gpu:int=0, 
        outdir:str=None, 
        projection:str=None,
        repeat:bool=False,
        impute:str=None, 
        chunk_size:int=20000,
        ignore_umap:bool=False,
        verbose:bool=False,
        assess:bool=False,
        show:bool=True,
        eval:bool=False,
        num_workers:int=4,
    ): # -> AnnData:
    """
    Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space
    
    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    profile
        Specify the single-cell profile, RNA or ATAC. Default: RNA.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    min_features
        Filtered out cells that are detected in less than min_features. Default: 600.
    min_cells
        Filtered out genes that are detected in less than min_cells. Default: 3.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. 
    batch_key
        Add the batch annotation to obs using this key. By default, batch_key='batch'.
    batch_size
        Number of samples per batch to load. Default: 64.
    lr
        Learning rate. Default: 2e-4.
    max_iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
    seed
        Random seed for torch and numpy. Default: 124.
    gpu
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: 'output/'.
    projection
        Use for new dataset projection. Input the folder containing the pre-trained model. If None, don't do projection. Default: None. 
    repeat
        Use with projection. If False, concatenate the reference and projection datasets for downstream analysis. If True, only use projection datasets. Default: False.
    impute
        If True, calculate the imputed gene expression and store it at adata.layers['impute']. Default: False.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    ignore_umap
        If True, do not perform UMAP for visualization and leiden for clustering. Default: False.
    verbose
        Verbosity, True or False. Default: False.
    assess
        If True, calculate the entropy_batch_mixing score and silhouette score to evaluate integration results. Default: False.
    
    Returns
    -------
    The output folder contains:
    adata.h5ad
        The AnnData matrice after batch effects removal. The low-dimensional representation of the data is stored at adata.obsm['latent'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    log.txt
        Records raw data information, filter conditions, model parameters etc.
    umap.pdf 
        UMAP plot for visualization.
    """
    
    np.random.seed(seed) # seed
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    if outdir:
        # outdir = outdir+'/'
        os.makedirs(os.path.join(outdir, 'checkpoint'), exist_ok=True)
        log = create_logger('scCobra', fh=os.path.join(outdir, 'log.txt'), overwrite=True)
    else:
        log = create_logger('scCobra')

    if not projection:
        adata, trainloader, testloader = load_data(
            data_list, batch_categories, 
            join=join,
            profile=profile,
            target_sum=target_sum,
            n_top_features=n_top_features,
            batch_size=batch_size, 
            chunk_size=chunk_size,
            min_features=min_features, 
            min_cells=min_cells,
            fraction=fraction,
            n_obs=n_obs,
            processed=processed,
            use_layer=use_layer,
            backed=backed,
            batch_name=batch_name, 
            batch_key=batch_key,
            keep_mt=keep_mt,
            log=log,
            num_workers=num_workers,
        )
        
        early_stopping = EarlyStopping(patience=10, checkpoint_file=os.path.join(outdir, 'checkpoint/model.pt') if outdir else None)
        x_dim = adata.shape[1] if (use_layer == 'X' or use_layer in adata.layers) else adata.obsm[use_layer].shape[1]
        n_domain = len(adata.obs['batch'].cat.categories)
        

        model = scCobra_model(x_dim, n_domain=n_domain)
        
        model.fit(
            trainloader, 
            lr=lr, 
            max_iteration=max_iteration, 
            device=device, 
            early_stopping=early_stopping, 
            verbose=verbose,
        )
        if outdir:
            torch.save({'x_dim':x_dim,'n_domain':n_domain}, os.path.join(outdir, 'checkpoint/config.pt'))
    else:
        state = torch.load(os.path.join(projection, 'checkpoint/config.pt'))
        x_dim, n_domain = state['x_dim'], state['n_domain']
        model = VAE(x_dim, n_domain=n_domain)
        model.load_model(os.path.join(projection, 'checkpoint/model.pt'))
        model.to(device)
        
        adata, trainloader, testloader = load_data(
            data_list, batch_categories,  
            join='outer', 
            profile=profile,
            target_sum=target_sum,
            chunk_size=chunk_size,
            n_top_features=n_top_features, 
            min_cells=0,
            min_features=min_features,
            processed=processed,
            batch_name=batch_name,
            batch_key=batch_key,
            log = log,
            num_workers=num_workers,
        )
#         log.info('Processed dataset shape: {}'.format(adata.shape))
        
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device, eval=eval) # save latent rep
    if impute:
        adata.layers['impute0'] = model.encodeBatch(testloader, out='impute', batch_id=impute[0], device=device, eval=eval)
    # log.info('Output dir: {}'.format(outdir))
    
    model.to('cpu')
    del model
    if projection and (not repeat):
        ref = sc.read_h5ad(os.path.join(projection, 'adata.h5ad'))
        adata = AnnData.concatenate(
            ref, adata, 
            batch_categories=['reference', 'query'], 
            batch_key='projection', 
            index_unique=None
        )

    if outdir is not None:
        adata.write(os.path.join(outdir, 'adata.h5ad'), compression='gzip')  

    if not ignore_umap: #and adata.shape[0]<1e6:
        log.info('Plot umap')
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
        sc.tl.umap(adata, min_dist=0.1)
        sc.tl.leiden(adata)
        adata.obsm['X_scCobra_umap'] = adata.obsm['X_umap']
        
        # UMAP visualization
        sc.set_figure_params(dpi=80, figsize=(3,3))
        cols = ['batch', 'celltype', 'cell_type', 'leiden']
        color = [c for c in cols if c in adata.obs]
        if outdir:
            sc.settings.figdir = outdir
            save = '.png'
        else:
            save = None

        if len(color) > 0:
            if projection and (not repeat):
                embedding(adata, color='leiden', groupby='projection', save=save, show=show)
            else:
                sc.pl.umap(adata, color=color, save=save, wspace=0.4, ncols=4, show=show)  
        if assess:
            if len(adata.obs['batch'].cat.categories) > 1:
                entropy_score = batch_entropy_mixing_score(adata.obsm['X_umap'], adata.obs['batch'])
                log.info('batch_entropy_mixing_score: {:.3f}'.format(entropy_score))

            if 'celltype' in adata.obs:
                sil_score = silhouette_score(adata.obsm['X_umap'], adata.obs['celltype'].cat.codes)
                log.info("silhouette_score: {:.3f}".format(sil_score))

    if outdir is not None:
        adata.write(os.path.join(outdir, 'adata.h5ad'), compression='gzip')
    
    return adata
        
        

def label_transfer(ref, query, rep='latent', label='celltype'):
    """
    Label transfer
    
    Parameters
    -----------
    ref
        reference containing the projected representations and labels
    query
        query data to transfer label
    rep
        representations to train the classifier. Default is `latent`
    label
        label name. Defautl is `celltype` stored in ref.obs
    
    Returns
    --------
    transfered label
    """

    from sklearn.neighbors import KNeighborsClassifier
    
    X_train = ref.obsm[rep]
    y_train = ref.obs[label]
    X_test = query.obsm[rep]
    
    knn = knn = KNeighborsClassifier().fit(X_train, y_train)
    y_test = knn.predict(X_test)
    
    return y_test

