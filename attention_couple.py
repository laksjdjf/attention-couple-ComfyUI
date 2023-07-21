import torch
import torch.nn.functional as F
import copy

def get_masks_from_q(masks, q):
    ret_masks = []
    for mask in masks:
        if isinstance(mask,torch.Tensor):
            mask_size = mask.shape[1] * mask.shape[2]
            down_sample_rate = int((mask_size // 64 // q.shape[1]) ** (1/2))
            mask_downsample = F.interpolate(mask.unsqueeze(0), scale_factor= 1/8/down_sample_rate, mode="nearest").squeeze(0)
            mask_downsample = mask_downsample.view(1,-1, 1).repeat(q.shape[0], 1, q.shape[2])
            ret_masks.append(mask_downsample)
        else: # coupling処理なしの場合
            ret_masks.append(torch.ones_like(q))
    return ret_masks

class AttentionCouple:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"], ),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode):
        if mode == "Latent":
            return (model, positive, negative) # latent coupleの場合は何もしない
        
        self.negative_positive_masks = []
        self.negative_positive_conds = []
        
        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)
        
        dtype = model.model.diffusion_model.dtype
        device = "cuda"
        
        # maskとcondをリストに格納する
        for conditions in [new_negative, new_positive]:
            conditions_masks = []
            conditions_conds = []
            if len(conditions) != 1:
                mask_norm = torch.stack([cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"] for cond in conditions])
                mask_norm = mask_norm / mask_norm.sum(dim=0) # 合計が1になるように正規化(他が0の場合mask_strengthの効果がなくなる)
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])
                conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])
                del conditions[0][1]["mask"] # latent coupleの無効化のため
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)
        self.conditioning_length = (len(new_negative), len(new_positive))
            
        new_model = copy.deepcopy(model)
        self.hook_forwards(new_model.model.diffusion_model)
        
        return (new_model, [new_positive[0]], [new_negative[0]]) # pool outputは・・・後回し
    
    def hook_forwards(self, root_module: torch.nn.Module):
        for name, module in root_module.named_modules():
            if "attn2" in name and "CrossAttention" in module.__class__.__name__:
                module.forward = self.hook_forward(module)
    
    def hook_forward(self, module):
        def forward(x, context=None, value=None, mask=None):
            '''
            uncond = [uncond1 * batch_size, uncond2 * batch_size, ...]
            cond = [cond1 * batch_size, cond2 * batch_size, ...]
            concat = [uncond, cond] (全てのサンプラーがこの順番なことを祈る)
            '''

            q = module.to_q(x)
            
            len_neg, len_pos = self.conditioning_length # negative, positiveの長さ
            q_uncond, q_cond = q.chunk(2) # uncond, condの分割
            b = q_cond.shape[0] # batch_size
            
            # maskの作成
            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_uncond)
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_cond)
            masks = torch.cat(masks_uncond + masks_cond)

            # qをconditionの数だけ拡張
            q_target= torch.cat([q_uncond]*len_neg + [q_cond]*len_pos, dim=0)
            q_target = q_target.view(b*(len_neg+len_pos), -1, module.heads, module.dim_head).transpose(1, 2)
            
            # contextをbatch_sizeだけ拡張
            context_uncond = torch.cat([cond.repeat(b,1,1) for cond in self.negative_positive_conds[0]], dim=0)
            context_cond = torch.cat([cond.repeat(b,1,1) for cond in self.negative_positive_conds[1]], dim=0)
            context = torch.cat([context_uncond, context_cond], dim=0)

            k = module.to_k(context)
            v = module.to_v(context)
            
            k = k.view(b * (len_neg + len_pos), -1, module.heads, module.dim_head).transpose(1, 2)
            v = v.view(b * (len_neg + len_pos), -1, module.heads, module.dim_head).transpose(1, 2)

            qkv = torch.nn.functional.scaled_dot_product_attention(q_target, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            qkv = qkv.transpose(1, 2).reshape(b * (len_neg + len_pos), -1, module.heads * module.dim_head)
            qkv = qkv * masks
            
            out_uncond = qkv[:len_neg*b].view(len_neg, b, -1, module.heads * module.dim_head).sum(dim=0)
            out_cond = qkv[len_neg*b:].view(len_pos, b, -1, module.heads * module.dim_head).sum(dim=0)
            out = torch.cat([out_uncond, out_cond], dim=0)
            return module.to_out(out)
        return forward
        
NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Load Attention couple",
}
