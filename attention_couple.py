import torch
import torch.nn.functional as F

def get_masks_from_q(masks, q):
    ret_masks = []
    for mask in masks:
        if isinstance(mask,torch.Tensor):
            mask_size = mask.shape[1] * mask.shape[2]
            down_sample_rate = int((mask_size // 64 // q.shape[1]) ** (1/2))
            mask_downsample = F.interpolate(mask.unsqueeze(0), scale_factor= 1/8/down_sample_rate, mode="nearest").squeeze(0)
            mask_downsample = mask_downsample.view(1,-1, 1).repeat(q.shape[0], 1, q.shape[2])
            mask_downsample = mask_downsample.to(q.device, dtype=q.dtype)
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
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative):
        self.negative_positive_masks = []
        self.negative_positive_conds = []

        # maskとcondをリストに格納する
        for conditions in [negative, positive]:
            conditions_masks = []
            conditions_conds = []
            if len(conditions) != 1:
                mask_norm = torch.stack([cond[1]["mask"] * cond[1]["mask_strength"] for cond in conditions])
                mask_norm = mask_norm / mask_norm.sum(dim=0) # 合計が1になるように正規化
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])
                conditions_conds.extend([cond[0] for cond in conditions])
                del conditions[0][1]["mask"] # latent coupleの無効化
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                conditions_conds = [conditions[0][0]]
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)

        self.hook_forwards(model.model.diffusion_model)
        
        return (model, [positive[0]], [negative[0]]) # pool outputは・・・後回し
    
    def hook_forwards(self, root_module: torch.nn.Module):
        for name, module in root_module.named_modules():
            if "attn2" in name and "CrossAttention" in module.__class__.__name__:
                module.forward = self.hook_forward(module)
    
    def hook_forward(self, module):
        def forward(x, context=None, value=None, mask=None):
            q = module.to_q(x)
            outs = []
            # negative, positiveの順に処理。
            for i in range(2):
                q_cond = q.chunk(2)[i]
                b = q_cond.shape[0]
                out = torch.zeros_like(q_cond)
                masks = get_masks_from_q(self.negative_positive_masks[i], q_cond)
                q_cond = q_cond.view(b, -1, module.heads, module.dim_head).transpose(1, 2)
                for mask, context in zip(masks, self.negative_positive_conds[i]):
                    k = module.to_k(context.to(q_cond.device, dtype=q_cond.dtype)).repeat(b, 1, 1)
                    v = module.to_v(context.to(q_cond.device, dtype=q_cond.dtype)).repeat(b, 1, 1)
                    
                    k = k.view(b, -1, module.heads, module.dim_head).transpose(1, 2)
                    v = v.view(b, -1, module.heads, module.dim_head).transpose(1, 2)

                    qkv = torch.nn.functional.scaled_dot_product_attention(q_cond, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
                    qkv = qkv.transpose(1, 2).reshape(b, -1, module.heads * module.dim_head)
                    out += qkv * mask
                outs.append(out)

            return module.to_out(torch.cat(outs)) # uncond, cond 全てのサンプラーがこの順番なことを祈る
        return forward
        
NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Load Attention couple",
}