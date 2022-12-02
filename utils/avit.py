import torch
import torch.nn as nn
from torch.autograd import Variable

import timm
from timm.models.vision_transformer import VisionTransformer

import wandb

class Avit(VisionTransformer):
    def __init__(self, net:VisionTransformer):
        super(Avit, self).__init__(img_size = net.default_cfg['input_size'][1], patch_size=16, in_chans=net.default_cfg['input_size'][0], num_classes = net.num_classes, embed_dim = net.embed_dim)
        self.num_classes = net.num_classes
        self.num_features = net.num_features
        self.num_tokens = 1
        self.num_heads = self.embed_dim // 64
        self.scale = 64 ** -0.5

        self.gate_scale = 10
        self.gate_center = 75

        num_patches = net.patch_embed.num_patches

        print('\nNow this is an ACT DeiT.\n')
        self.eps = 0.01
        print(f'Setting eps as {self.eps}.')

        print('Now setting up the rho.')
        self.rho = None  # Ponder cost
        self.counter = None  # Keeps track of how many layers are used for each example (for logging)
        self.batch_cnt = 0 # amount of batches seen, mainly for tensorboard

        # for token act part
        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.counter_token = None
        self.total_token_cnt = num_patches + self.num_tokens

        self.step = 0

    def forward_mask_attn(self, net, x, mask, mask_softmax_bias = -1000.):
        B, N, C = x.shape
        qkv = net.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #! qkv: [3, 10, 3, 197, 64]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) #! [10, 3, 197, 64]
        attn = net.matmul1(q, k.transpose(-2, -1)) * self.scale #! [10, 3, 197, 197] 3:heads

        if mask is not None:
            # now we need to mask out all the attentions associated with this token
            attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * mask_softmax_bias #! [10, 3, 197, 197] + [10, 1, 1, 197]
            # this additional bias will make attention associated with this token to be zeroed out
            # this incurs at each head, making sure all embedding sections of other tokens ignore these tokens

        attn = attn.softmax(dim=-1) #! [10, 3, 197, 197]
        attn = net.attn_drop(attn)

        x = net.matmul2(attn, v).transpose(1, 2).reshape(B, N, C) #! [10, 197, 192]
        x = net.proj(x)
        x = net.proj_drop(x)

        return x
        

    def forward_act_block(self, net, x, mask = None):
        bs, token, dim = x.shape

        if mask is None:
            x = x + net.attn(net.norm1(x))
            x = x + net.mlp(net.norm2(x))
        else:
            x = x + self.forward_mask_attn(net.attn, net.norm1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1), mask=mask) #! every step need to load the mask!!
            x = x + net.mlp(net.norm2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1))

        halting_score_token = torch.sigmoid(self.gate_scale * x[:,:,0] - self.gate_center)
        halting_score = [-1, halting_score_token]

        return x, halting_score

    def forward_act(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1) #! [10, 197, 192]
        x = self.pos_drop(x + self.pos_embed) #! pos_embed: [1, 197, 192]
        
        # now start the act part
        bs = x.size()[0]  # The batch size

        # this part needs to be modified for higher GPU utilization
        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.R_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.mask_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.counter_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

        c_token = self.c_token.clone() #! [10, 197] {line 2}
        R_token = self.R_token.clone() #! [10, 197] Remainder value {line 3}
        mask_token = self.mask_token.clone() #! [10, 197] Token mask {line 6}
        self.rho_token = self.rho_token.detach() * 0. #! Token ponder loss vector {line 5}
        self.counter_token = self.counter_token.detach() * 0 + 1. #! [10, 197]
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = None
        # Use out to backbone
        out = x

        self.halting_score_layer = []

        for i, l in enumerate(self.blocks):
            # block out all the parts that are not used
            out.data = out.data * mask_token.float().view(bs, self.total_token_cnt, 1) #! [10, 197, 192] * [10, 197, 1] {line 8}

            # evaluate layer and get halting probability for each sample
            # block_output, h_lst = l.forward_act(out)    # h is a vector of length bs, block_output a 3D tensor
            block_output, h_lst = self.forward_act_block(l, out, 1.- mask_token.float()) #! h_lst: [-1, [10, 197]]   # h is a vector of length bs, block_output a 3D tensor

            # if self.args.distr_prior_alpha > 0.: #! calculate the halting score distribution in eq.11
            self.halting_score_layer.append(torch.mean(h_lst[1][1:])) #? the mean of [9, 197] 

            out = block_output.clone()              # Deep copy needed for the next layer

            _, h_token = h_lst # h is layer_halting score, h_token is token halting score, first position discarded

            # here, 1 is remaining, 0 is blocked
            block_output = block_output * mask_token.float().view(bs, self.total_token_cnt, 1) #! [10, 197] -> [10, 197, 1]

            # Is this the last layer in the block?
            if i == len(self.blocks) - 1:
                h_token = Variable(torch.ones(bs, self.total_token_cnt).cuda()) #! {line 12}

            # for token part
            c_token = c_token + h_token #! {line 14}
            self.rho_token = self.rho_token + mask_token.float() #! {line 15}

            # Case 1: threshold reached in this iteration
            # token part
            reached_token = c_token > 1 - self.eps #! {line 17} #! [10, 197]
            if self.step % 10 == 0:
                print(f'avg_val_{i}', torch.mean(c_token).item())
                wandb.log({f'avg_val_{i}': torch.mean(c_token).item()})
                print(f"reached_token_ratio_{i}", torch.mean(reached_token.float()).item())
                wandb.log({f"reached_token_ratio_{i}": torch.mean(reached_token.float()).item()})
            reached_token = reached_token.float() * mask_token.float() #! 抽取出本轮达到目标值的token，同时还要忽略掉之前被mask掉的token
            delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1) #! {line 26}
            self.rho_token = self.rho_token + R_token * reached_token #! {line 20}

            # Case 2: threshold not reached
            # token part
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float() #! the masked token is directy included in the range
            R_token = R_token - (not_reached_token.float() * h_token) #! {line 18}
            delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1) #! {line 24}

            self.counter_token = self.counter_token + not_reached_token # These data points will need at least one more layer

            # Update the mask
            mask_token = c_token < 1 - self.eps #! {line 28}

            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

        x = self.norm(output)
        

        return x[:, 0] #! [10, 192]

    def forward(self, x):
        x = self.forward_act(x)
        x = self.head(x)
        self.step += 1
        return x


if __name__ == '__main__':
    deit = timm.models.create_model('deit_base_patch16_224', pretrained=True)
    print(deit)
    avit = Avit(deit)
    print(avit.state_dict().keys())