import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConstantInput(nn.Module):
    def __init__(self,pe_dim):
        super(ConstantInput, self).__init__()
        self.const_value = nn.Parameter(torch.rand((1, 1, pe_dim)))

    def forward(self,x):
        return self.const_value


class SinActivation(nn.Module):
    def __init__(self,w=1):
        self.w=w
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(self.w * x)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_nerf_embedder(multires=6,input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim
    

class ResBlockFC(nn.Module):
    def __init__(self,in_dim,out_dim,hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim=out_dim
        self.fc1=nn.Linear(in_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,out_dim)

        if in_dim!=out_dim:
            self.shot_cut=nn.Linear(in_dim,out_dim)
        else:
            self.shot_cut=None
        self.activation = nn.LeakyReLU(0.2,inplace=True)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc1.weight, a=0.2, mode="fan_in")
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.zeros_(self.fc2.weight)

    def forward(self,x,z=0):
        if self.shot_cut is not None:
            identity=self.shot_cut(x)
        else:
            identity=x
        out= self.fc1(self.activation(x+z))
        out=self.fc2(self.activation(out))
        out = out+identity
        return out

class MLP(nn.Module):
    def __init__(self,x_dim,z_dim,
                hidden_dim_1,hidden_dim_2,
                n_layers_stage1=3,
                n_layers_stage2=3,):
        super().__init__()
        self.first_fc=nn.Linear(x_dim,hidden_dim_1)
        self.activation= nn.LeakyReLU(0.2,inplace=True)
        self.blocks=nn.ModuleList()
        self.c1_projs_offset=nn.ModuleList()
        self.c2_projs_offset=nn.ModuleList()
        for i in range(n_layers_stage1):
            self.blocks.append(ResBlockFC(hidden_dim_1,hidden_dim_1))
            self.c1_projs_offset.append(nn.Linear(z_dim,hidden_dim_1))
        self.blocks_stage2=nn.ModuleList()
        for i in range(n_layers_stage2):
            self.blocks_stage2.append(ResBlockFC(hidden_dim_1 if i==0 else hidden_dim_2,hidden_dim_2))
        self.last_fc=nn.Linear(hidden_dim_2,3)
        nn.init.kaiming_normal_(self.first_fc.weight, a=0.2, mode="fan_in")
        nn.init.constant_(self.first_fc.bias, 0.0)

    def forward(self,x,z,view_dim=1):
        # x z in format [bs,n_ref,K,ch]
        # constraint [bs,n_ref,K, ch]
        #print(x.shape)
        #print(self.first_fc.weight.shape)
        #print(z.shape)
        out=self.first_fc(x)
        for i in range(len(self.blocks)):
            out=self.blocks[i](out,self.c1_projs_offset[i](z))
        out=torch.mean(out,dim=view_dim)
        for i in range(len(self.blocks_stage2)):
            out=self.blocks_stage2[i](out)
        out=self.activation(out)
        out=self.last_fc(out)
        return out


class ImplicitDecoder(nn.Module):
    def __init__(
        self,
        x_dim,
        embedder="fixed_sincos",
        pe_dim=128, #positional embedding dim, valid for learnable positional embedding
        z_dim=128, #style dim
        hidden_dim_1=128,
        hidden_dim_2=128,
        constraint_dim=-1,
        n_layers_stage1=3,
        n_layers_stage2=3,
        ):
        super().__init__()
        if embedder=='fixed_sincos':
            self.positional_embedding,pe_dim=get_nerf_embedder(10,x_dim)
        elif embedder=='cips_learnable':
            pe_proj = nn.Linear(x_dim, pe_dim,bias=True)
            nn.init.uniform_(pe_proj.weight, -np.sqrt(9 / x_dim), np.sqrt(9 / x_dim))
            nn.init.constant_(pe_proj.bias,0)
            self.positional_embedding=nn.Sequential(pe_proj,SinActivation(w=1))
        elif embedder=='constant_learnable':
            self.positional_embedding=ConstantInput(pe_dim)
        elif embedder=='none':
            self.positional_embedding=nn.Identity()
            pe_dim=x_dim
        else:
            raise NotImplementedError('unknown positional embedding for p_3d')
        self.mlp=MLP(pe_dim,z_dim,hidden_dim_1,hidden_dim_2,n_layers_stage1,n_layers_stage2)

    def forward(self,x,z):
        """[summary]

        Args:
            x ([type]): [bs*nrender,K,3]
            z ([type]): [bs*nrender,n_ref,K,ch]
        """
        pe=self.positional_embedding(x) 
        if pe.dim()==3:
            pe=pe.unsqueeze(1) #[bs*nrender,1,K,ch]
        #print(pe.shape,self.mlp.first_fc.weight.shape)
        colors=self.mlp(pe,z)
        colors=torch.sigmoid(colors)
        return colors


