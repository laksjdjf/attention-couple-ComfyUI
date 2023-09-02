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

def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    to["patches_replace"]["attn2"][key] = patch

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

        new_model = model.clone()
        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")
        if not self.sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[0].attn2), ("input", id))
            set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2), ("middle", id))
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[0].attn2), ("output", id))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[index].attn2), ("input", id, index))
            for index in range(10):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[index].attn2), ("middle", id, index))
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[index].attn2), ("output", id, index))
        
        return (new_model, [new_positive[0]], [new_negative[0]]) # pool outputは・・・後回し
    
    def make_patch(self, module):
        def patch(q, k, v, extra_options):
            '''
            uncond = [uncond1 * batch_size, uncond2 * batch_size, ...]
            cond = [cond1 * batch_size, cond2 * batch_size, ...]
            concat = [uncond, cond] (全てのサンプラーがこの順番なことを祈る)
            '''
            
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
            return out
        return patch
        
NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Load Attention couple",
}
