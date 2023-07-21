# attention-couple-ComfyUI
プロンプトの領域指定を行うカスタムノードです。ComfyUIにはlatent coupleベースの領域指定法が存在しますが、こちらはUNet内のAttention層で領域分けします。

# 使い方
1. custom_nodesにくろーん
2. latent coupleと同様のノードを作る。参考：https://comfyui.creamlab.net/nodes/ConditioningSetMask
3. loaderからattention coupleのノードをロードしてMODELとCONDITIONINGを繋げる
4. 画像を作る

[attention_couple.json](https://github.com/laksjdjf/attention-couple-ComfyUI/blob/main/attention_couple.json)も参考にしてください。

比較用として、modeをLatentにすると、latent coupleになります。といってもこのノードで何かするわけではなく入力をそのまま出力するモードになるだけです。

# 注意点
実験段階でたぶん色々問題があります。

1. モデルのモジュールを無理やり書き換えるので他の機能になんらかの影響を与える可能性があります。
2. Pytorch 2.0以上が必要です。
3. サンプラーによっては想定しない動作が起こる可能性があります。（uncondとcondの順番が分からない・・・）
4. どのプロンプトにも指定されていない領域があるとエラーが起こります。
5. LoRAの領域指定は実装していないし見当もつきません。

# 参考
https://note.com/gcem156/n/nb3d516e376d7
