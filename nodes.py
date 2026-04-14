"""
Stretch Ref RoPE — Qwen Image Edit 高分辨率 Pose 修复
=====================================================
通过 ComfyUI 的 post_input patch 机制，在 ref_latents 坐标拼接完成后、
RoPE embedding 计算前，对指定 ref_latent 的位置坐标进行线性拉伸，
使其覆盖范围与输出 latent 对齐。

不修改任何 ComfyUI 核心代码。

原理：
  model.py _forward 中的执行顺序：
    1. process_img(x)          → 输出 latent 的 img_ids (num_embeds tokens)
    2. process_img(ref) × N    → ref_latents 的 kontext_ids 拼接到 img_ids
    3. "post_input" patches    → ★ 我们在这里修改 img_ids ★
    4. pe_embedder(ids)        → 计算 RoPE positional embedding

用法：
  Model Loader → StretchRefRoPE → KSampler

  stretch_indices: 指定要拉伸的 ref_latent 索引（从 1 开始，对应 image1/image2/image3）
  例如 pose 图在 image1 → 填 "1"
  例如 pose 图在 image3 → 填 "3"
  多个用逗号分隔 → "1,3"
  填 "0" 或留空 → 拉伸全部
"""

import torch

# Debug counter - only print first few times to avoid log spam
_debug_counter = 0
_DEBUG_MAX = 3


class StretchRefRoPE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "stretch_indices": ("STRING", {
                    "default": "1",
                    "tooltip": (
                        "要拉伸的 ref_latent 索引，从 1 开始，对应 TextEncodeQwenImageEditPlus "
                        "的 image1/image2/image3 顺序（跳过未连接的）。"
                        "多个用逗号分隔，如 \"1,3\"。填 \"0\" 或留空则拉伸全部。"
                    ),
                }),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "启用/禁用 RoPE 坐标拉伸"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"
    DESCRIPTION = (
        "将指定 ref_latent 的 RoPE 位置坐标线性拉伸到与输出 latent 相同的范围。"
        "用于解决高分辨率生成时 OpenPose 引导人物偏向中心的问题，"
        "同时保持 ref_latent 在 1024² 编码（4096 tokens），不增加计算量。"
        "通过 stretch_indices 可精确控制只拉伸 pose 图，不影响其他参考图。"
    )

    def apply(self, model, stretch_indices="1", enabled=True):
        if not enabled:
            return (model,)

        # 解析要拉伸的索引（1-based → 0-based）
        indices_to_stretch = set()
        stretch_all = False
        stripped = stretch_indices.strip()
        if stripped == "" or stripped == "0":
            stretch_all = True
        else:
            for part in stripped.split(","):
                part = part.strip()
                if part.isdigit() and int(part) > 0:
                    indices_to_stretch.add(int(part) - 1)  # 转成 0-based

        model_patched = model.clone()

        # 确认 apply 被调用，以及 patches 写入成功
        print(f"[StretchRefRoPE] apply() called: stretch_all={stretch_all}, indices_to_stretch={indices_to_stretch}")
        print(f"[StretchRefRoPE] model_options keys: {list(model_patched.model_options.keys())}")
        to = model_patched.model_options.get("transformer_options", {})
        print(f"[StretchRefRoPE] transformer_options keys before patch: {list(to.keys())}")
        if "patches" in to:
            print(f"[StretchRefRoPE] existing patches: {list(to['patches'].keys())}")

        def stretch_ref_rope_patch(patch_input):
            """post_input patch: 在 PE embedding 计算前修改 ref 部分的坐标"""
            global _debug_counter
            img = patch_input["img"]
            txt = patch_input["txt"]
            img_ids = patch_input["img_ids"]
            txt_ids = patch_input["txt_ids"]
            transformer_options = patch_input["transformer_options"]

            should_log = _debug_counter < _DEBUG_MAX

            ref_num_tokens = transformer_options.get("reference_image_num_tokens", None)
            if ref_num_tokens is None or len(ref_num_tokens) == 0:
                if should_log:
                    print("[StretchRefRoPE] patch called but NO ref_num_tokens found, skipping")
                    _debug_counter += 1
                return {"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids}

            total_ref_tokens = sum(ref_num_tokens)
            num_embeds = img_ids.shape[1] - total_ref_tokens

            if should_log:
                print(f"[StretchRefRoPE] patch called: img_ids shape={img_ids.shape}, "
                      f"ref_num_tokens={ref_num_tokens}, num_embeds={num_embeds}, "
                      f"indices_to_stretch={indices_to_stretch}, stretch_all={stretch_all}")

            if num_embeds <= 0:
                return {"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids}

            # 获取输出 latent 的 h/w 坐标范围
            out_h_coords = img_ids[:, :num_embeds, 1]
            out_w_coords = img_ids[:, :num_embeds, 2]
            out_h_min = out_h_coords.min().item()
            out_h_max = out_h_coords.max().item()
            out_w_min = out_w_coords.min().item()
            out_w_max = out_w_coords.max().item()
            out_h_span = out_h_max - out_h_min
            out_w_span = out_w_max - out_w_min

            if should_log:
                print(f"[StretchRefRoPE] output coords: h=[{out_h_min:.1f}, {out_h_max:.1f}] (span={out_h_span:.1f}), "
                      f"w=[{out_w_min:.1f}, {out_w_max:.1f}] (span={out_w_span:.1f})")

            if out_h_span == 0 or out_w_span == 0:
                return {"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids}

            # 逐个 ref 处理，只拉伸指定的索引
            stretched_any = False
            ref_start = num_embeds
            for ref_idx, ref_tokens in enumerate(ref_num_tokens):
                ref_end = ref_start + ref_tokens

                should_stretch = stretch_all or (ref_idx in indices_to_stretch)
                if not should_stretch:
                    if should_log:
                        print(f"[StretchRefRoPE] ref[{ref_idx}]: {ref_tokens} tokens — SKIP (not in indices)")
                    ref_start = ref_end
                    continue

                ref_h = img_ids[:, ref_start:ref_end, 1]
                ref_w = img_ids[:, ref_start:ref_end, 2]
                ref_h_min = ref_h.min().item()
                ref_h_max = ref_h.max().item()
                ref_w_min = ref_w.min().item()
                ref_w_max = ref_w.max().item()
                ref_h_span = ref_h_max - ref_h_min
                ref_w_span = ref_w_max - ref_w_min

                if should_log:
                    print(f"[StretchRefRoPE] ref[{ref_idx}]: {ref_tokens} tokens, "
                          f"h=[{ref_h_min:.1f}, {ref_h_max:.1f}] (span={ref_h_span:.1f}), "
                          f"w=[{ref_w_min:.1f}, {ref_w_max:.1f}] (span={ref_w_span:.1f})")

                # 只在 ref 范围小于输出范围时拉伸
                if ref_h_span > 0 and ref_h_span < out_h_span:
                    h_scale = out_h_span / ref_h_span
                    ref_h_center = (ref_h_min + ref_h_max) / 2.0
                    img_ids[:, ref_start:ref_end, 1] = (ref_h - ref_h_center) * h_scale + ref_h_center
                    stretched_any = True
                    if should_log:
                        new_h_min = img_ids[:, ref_start:ref_end, 1].min().item()
                        new_h_max = img_ids[:, ref_start:ref_end, 1].max().item()
                        print(f"[StretchRefRoPE]   h stretched: scale={h_scale:.2f}, "
                              f"new h=[{new_h_min:.1f}, {new_h_max:.1f}]")

                if ref_w_span > 0 and ref_w_span < out_w_span:
                    w_scale = out_w_span / ref_w_span
                    ref_w_center = (ref_w_min + ref_w_max) / 2.0
                    img_ids[:, ref_start:ref_end, 2] = (ref_w - ref_w_center) * w_scale + ref_w_center
                    stretched_any = True
                    if should_log:
                        new_w_min = img_ids[:, ref_start:ref_end, 2].min().item()
                        new_w_max = img_ids[:, ref_start:ref_end, 2].max().item()
                        print(f"[StretchRefRoPE]   w stretched: scale={w_scale:.2f}, "
                              f"new w=[{new_w_min:.1f}, {new_w_max:.1f}]")

                if should_log and not stretched_any:
                    print(f"[StretchRefRoPE]   ref[{ref_idx}]: NO stretch needed "
                          f"(ref span >= output span)")

                ref_start = ref_end

            if should_log:
                _debug_counter += 1

            return {"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids}

        model_patched.set_model_patch(stretch_ref_rope_patch, "post_input")

        # 确认 patch 写入成功
        to_after = model_patched.model_options.get("transformer_options", {})
        patches_after = to_after.get("patches", {})
        print(f"[StretchRefRoPE] patches after set_model_patch: {list(patches_after.keys())}")
        if "post_input" in patches_after:
            print(f"[StretchRefRoPE] post_input has {len(patches_after['post_input'])} function(s)")
            print(f"[StretchRefRoPE] post_input func id: {id(patches_after['post_input'][0])}")
            print(f"[StretchRefRoPE] model_options id: {id(model_patched.model_options)}")
            print(f"[StretchRefRoPE] transformer_options id: {id(to_after)}")
            print(f"[StretchRefRoPE] patches dict id: {id(patches_after)}")

        # 额外：用 model_function_wrapper 拦截，看 transformer_options 内容
        orig_wrapper = model_patched.model_options.get("model_function_wrapper", None)
        _wrapper_counter = [0]
        def debug_model_wrapper(apply_model_func, args):
            if _wrapper_counter[0] < 2:
                c = args.get("c", {})
                to = c.get("transformer_options", {})
                p = to.get("patches", {})
                print(f"[StretchRefRoPE] === model_function_wrapper called ===")
                print(f"[StretchRefRoPE] transformer_options keys: {list(to.keys())}")
                print(f"[StretchRefRoPE] patches keys: {list(p.keys())}")
                if "post_input" in p:
                    print(f"[StretchRefRoPE] post_input count: {len(p['post_input'])}")
                else:
                    print(f"[StretchRefRoPE] !!! post_input NOT in patches !!!")
                _wrapper_counter[0] += 1
            return apply_model_func(args["input"], args["timestep"], **args["c"])
        model_patched.model_options["model_function_wrapper"] = debug_model_wrapper

        return (model_patched,)


NODE_CLASS_MAPPINGS = {
    "StretchRefRoPE": StretchRefRoPE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StretchRefRoPE": "Stretch Ref RoPE (Qwen Image Edit)",
}
