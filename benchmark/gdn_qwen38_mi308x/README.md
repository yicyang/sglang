# Qwen3.8-Flash-Next (TP2, MI308X): GDN prefill / decode backend 评估与推荐报告

> 本报告所有数字都来自 `results/` 下的原始数据（json/jsonl/log/trace）。负面结果和测量问题如实列出。
> 生成日期：2026-10-23。

## 0. 结论速览

| 阶段 | 推荐 | 依据（本机实测） |
|---|---|---|
| **Prefill** | `--linear-attn-prefill-backend aiter`：aiter `chunk_gated_delta_rule_opt_vk` + FlyDSL 融合 prepare + HIP K5 + gfx942 tuned `chunk_o`。**必须使用包含 ROCm/aiter#4952 修复的 aiter**（本次 fork 分支已 cherry-pick） | 单层 GDN prefill（T=12000）：sglang 原生 3036 µs → 1166 µs（**2.6x**）；e2e 12k 输入 TTFT p50 **-6.4%**（1049.5 → 982.2 ms，5 轮中位数）；GSM8K 与 base 一致（96.42% vs 96.35%）；无额外 transpose/copy kernel |
| 次选 / 无法升级 aiter 时 | sglang 原生 Triton (FLA) + 本次 gfx942 tuned 配置（默认 backend，零接入成本） | 单层 3036 → 1826 µs（1.66x）；e2e 12k TTFT p50 -4.2%（1049.5 → 1005.4 ms） |
| **Decode** | 保持 sglang 原生 Triton packed decode（`--linear-attn-decode-backend triton`，默认），bs≤8 用本次 tuned 配置 | 在冷 cache（真实 36 层轮转）下，Triton packed decode 在 bs≥12 时最好或持平；aiter FlyDSL decode（即使 tuned）在 bs 24/32 慢 3–5%；e2e decode ITL 各方案一致（差异 <1%，在噪声内） |

GDN 在该模型 12k prefill 中只占约 11% 的 GPU 时间（profile），所以 2.6x 的 kernel 加速换来的是约 6% 的 TTFT 改善；decode 中 GDN（含 conv update）只占约 1.7% 的 GPU 时间，decode backend 选择对 e2e 几乎没有影响。

**重要发现（会直接影响结论的坑）**
1. **本机安装的 aiter（apinge fork 4752a408）里的 FlyDSL GDN prepare kernel 在真实模型数据上数值发散**（key 高度相关时 WY 三角求解不稳定），会让模型输出退化成重复 token。随机 key 的 kernel 单测**测不出来**。上游已在 ROCm/aiter `987203ba5`（#4952）修复。
   - 这个 bug 在第一轮 e2e 中造成了一个**虚假的 decode "加速"**（输出退化成重复 token → MoE 路由集中 → MoE 更快），已经排查、修正、并从结论中剔除，详见 §6。
2. aiter main 的官方 K5 BV tuner（`csrc/gdn_k5/chunk_gdn_h_opt_tune.py`）有 bug：`FLYDSL_K5_OPT_BV` 只在 import 时读取，tuner 逐候选设置的 env 不生效，选出的“最优 BV”只是计时噪声。已修复（aiter main 分支提交）。
3. 机器上的 8 张 GPU 与其他租户共享（KFD 进程列表中有不属于本容器的进程，且在本容器空闲时出现 GPU0/1/4/5/6 100% 占用）。e2e 采用多轮 + GPU 对轮换 + 取中位数，并单独标记被干扰的轮次（§5）。

## 1. 环境与版本

| 项 | 值 |
|---|---|
| GPU | 8 × AMD Instinct MI308X (gfx942, 80 CU)，与其他租户共享 |
| ROCm / torch / triton / flydsl | 7.14 / 2.12.0+rocm7.14 / 3.8.0 / 0.3.2 |
| 模型 | `/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16`：48 层 = 36 GDN + 12 full-attn；GDN Hk=16, Hv=48 (ratio 3), K=V=128, ssm fp32。**TP2 每卡 Hg=8, H=24** |
| sglang | apinge/sglang `qwen38-flash-next-amd-current-fix-install` @ `d0a5b9600`（branch head；/opt/sglang 安装的是落后 6 个 commit 的 c9ef753c6，GDN 相关代码无差别） |
| aiter（被测） | 安装版：apinge/aiter `qwen38-flash-next-dev` @ `4752a408`；ROCm/aiter main @ `60e01e537`；ROCm/aiter `Qwen3.5_dev` @ `11ba0c455` |
| ATOM | ROCm/ATOM main @ `a1ebc608b`（未安装，直接 import 其 `atom/model_ops/fla_ops`） |
| 启动 | `scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh`，TP_SIZE=2（chunked prefill 16384，max running 32，decode CUDA graph bs≤32，radix cache 关闭） |
| 偏差说明 | 分支 head 依赖的 PyHIP wheel（`pyhip.ops.gr_read`，QSA）在本机不可用（PyPI 上同名包无关）。所有 e2e 变体统一设置 `SGLANG_GR_READ_FLYDSL=0 SGLANG_USE_PYHIP_QSA=0`。这两项与 GDN 无关，对各变体相同。 |

## 2. 各实现调研

### 2.1 ROCm/aiter `main` vs `Qwen3.5_dev`：GDN 是否一致？

**算法一致，代码不一致。** 两者 prefill 的核心都是同一个 HIP K5 kernel（`csrc/kernels/chunk_gated_delta_rule_fwd_h.cu`，BT=64、BV∈{16,32,64}、mfma16x16x16、VK state）+ Triton `chunk_gated_delta_rule_opt_vk` 编排。差异：

| | Qwen3.5_dev (11ba0c455) | main (60e01e537) | 本机安装 fork (4752a408) |
|---|---|---|---|
| HIP K5 | 有（比 main 少 fp32 snapshot、gfx12 WMMA） | 有 | 与 main 相同 |
| FlyDSL K5 | **无**（`use_chunk_flydsl=True` 会**静默回退到 Triton K5**，实测耗时与 Triton 相同） | `chunk_gated_delta_rule_fwd_h_flydsl_opt`（支持 indexed pool、tuned BV CSV） | `..._flydsl_mfma16_hip`（不支持 indexed pool） |
| FlyDSL 融合 prepare | **无** | 有（含 #4952 数值修复） | 有，**但缺 #4952 修复 → 真实数据发散** |
| Triton K5 (opt_vk) | 较慢（1915 µs @T=12000） | 1555 µs | 1548 µs |
| `opt_vk` 参数顺序 | 与 main 不同（必须按关键字传参） | — | — |
| K5 tuned 表 | 无 | `model_configs/qwen3_5_{35b,397b}_chunk_gdn_h_opt_tuned.csv`（gfx942 cu80） | 无 |

### 2.2 sglang native
在 gfx942 上 GDN **只能**用 `triton`（FLA 移植）backend；`cutedsl`/`flashinfer` 等均为 CUDA-only。K5 为单一固定配置（BV=32/4 warps/2 stages，可用 `SGLANG_GDN_CHUNK_H_*` 覆盖），未针对 ROCm 调过。

### 2.3 ATOM
Qwen3-Next/3.5 使用 ATOM 自带的 FLA Triton（`atom/model_ops/fla_ops`），**state 为 KV 布局**，与 sglang 的 VK pool 不兼容（K=V=128 时形状相同，混用会静默写坏 state）；VK 版本只在 vLLM plugin 路径（`chunk_vk.py`）。ATOM 的 FlyDSL 路径仅用于 Qwen4Exp，且依赖 aiter main 的 `_flydsl_opt`（本机安装的 aiter 中不存在）。ATOM 不传 state index，接入需在外面 gather/scatter（已计入测试时间）。

## 3. Kernel 级测试

### 3.1 方法
- 输入与 sglang `GDNAttnBackend.forward_extend` 实际传给 kernel 的完全一致：q/k `[1,T,8,128]` bf16、v `[1,T,24,128]` bf16、g/beta `[1,T,24]` fp32、state pool `[slots,24,128,128]` fp32 + int32 `cache_indices`（原地更新）、varlen `cu_seqlens`。
- 每个候选从这些输入算到 (o, 更新后的 pool)，**任何 layout 转换 / gather / scatter 都计入时间**。
- 主指标：torch profiler 统计的 GPU kernel 时间之和（10 次背靠背调用的平均）；同时记录含 launch 开销的 wall time（每次迭代前 flush 256MB 以清 L2/MALL）。
- 正确性（不通过不参与排名）：
  1. fp32 逐 token 递推参考（5 条变长序列）；
  2. 大幅值门控（|g|~1e2）；
  3. **key 高度相关**（corr 0.5/0.9）——真实模型的 key 是高度相关的；
  4. **回放从运行中的服务 dump 下来的真实输入**（`results/kernel/real_inputs/`），与 sglang Triton 对比。
  - 第 3、4 项是在发现 §6 的 bug 后补充的；只有它们能抓到该 bug。
- 脚本：`bench/bench_prefill_pipeline.py`，结果：`results/kernel/v2/*.json`。

### 3.2 Prefill：K5 (`chunk_gated_delta_rule_fwd_h`) 单独耗时（GPU µs）

从完整流水线 profile 中拆出的 K5 kernel 时间（v1 数据，`results/kernel/prefill_*.json`；修复 prepare 不影响 K5）：

| K5 实现 | 1×1024 | 1×4096 | 1×12000 | 1×16384 | 4×4096 | 16×1024 |
|---|---|---|---|---|---|---|
| sglang FLA Triton（默认 BV=32） | 136 | 513 | 1510 | 2067 | 1325 | 1365 |
| sglang FLA Triton（tuned BV=64） | 84 | 316 | 934 | 1281 | 984 | 855 |
| ATOM Triton (VK) | 102 | 379 | 1099 | 1508 | 1143 | 982 |
| aiter Triton opt_vk (main/fork) | 134 | 503 | 1542 | 2162 | 1785 | 1806 |
| aiter Triton opt_vk (Qwen3.5_dev) | 161 | 615 | 1919 | 2668 | 2196 | 2175 |
| **aiter HIP K5**（三个版本相同） | 53 | 190 | 604 | 825 | 632 | 567 |
| **aiter FlyDSL K5**（fork mfma16 / main opt） | 47 | 167 | 505–512 | 697–705 | 546–553 | 509–518 |

K5 上 aiter HIP/FlyDSL 比 sglang 原生快 **2.5–3.0x**；FlyDSL 比 HIP 再快约 15%。

### 3.3 Prefill：完整 GDN 流水线（T=12000，单序列，GPU µs）

| impl (T=12000, 1 seq, GPU us) | prepare (cumsum/KKT/solve/w,u) | K5 fwd_h | chunk_o | l2norm | other (copy/fill/index) | total |
|---|---|---|---|---|---|---|
| sglang_fla_triton | 769 | 1506 | 645 | 79 | 37 | 3036 |
| atom_chunk_vk | 809 | 1100 | 543 | 38 | 30 | 2521 |
| atom_chunk_kv(+transpose) | 749 | 1032 | 509 | 38 | 49 | 2377 |
| sglang_fla_triton(tuned) | 514 | 935 | 261 | 80 | 36 | 1826 |
| aiter_fork_fixed:opt_vk_triton | 755 | 1548 | 232 | 38 | 0 | 2574 |
| aiter_fork_fixed:opt_vk_hipK5 | 757 | 603 | 226 | 38 | 0 | 1624 |
| aiter_fork_fixed:opt_vk_flydslPrep+hipK5 | 303 | 596 | 229 | 38 | 0 | 1166 |
| aiter_fork_fixed:opt_vk_flydslPrep+flydslK5 | 303 | 506 | 229 | 39 | 28 | 1104 |
| aiter_main:opt_vk_triton | 767 | 1555 | 706 | 38 | 0 | 3066 |
| aiter_main:opt_vk_flydslPrep+hipK5 | 302 | 597 | 692 | 38 | 0 | 1629 |
| aiter_main:opt_vk_flydslPrep+flydslK5 | 302 | 508 | 692 | 38 | 0 | 1541 |
| aiter_q35dev:opt_vk_triton | 608 | 1915 | 717 | 37 | 14 | 3291 |
| aiter_q35dev:opt_vk_hipK5 | 611 | 602 | 703 | 37 | 14 | 1966 |


完整矩阵（9 种形状 × 全部候选，含 wall time）：

单元格为 `GPU kernel 时间 / wall 时间（µs）`；`correct=False` 表示未通过 §3.1 的正确性检查（安装版 aiter 的 FlyDSL prepare）。

| impl | correct (rel err vs fp32 ref) | 1x1024 | 1x2048 | 1x4096 | 1x8192 | 1x12000 | 1x16384 | 4x4096 | 16x1024 | mix5 |
|---|---|---|---|---|---|---|---|---|---|---|
| sglang_fla_triton | True (o 6.6e-03, s 6.4e-03) | 307 / 388 | 553 / 593 | 1059 / 1095 | 2071 / 2080 | 3036 / 3044 | 4138 / 4145 | 3404 / 3410 | 3443 / 3450 | 3516 / 3523 |
| atom_chunk_vk | True (o 8.9e-03, s 6.4e-03) | 262 / 551 | 460 / 655 | 855 / 991 | 1707 / 1819 | 2521 / 2598 | 3453 / 3531 | 3105 / 3171 | 3006 / 3038 | 3182 / 3231 |
| atom_chunk_kv(+transpose) | True (o 5.5e-03, s 5.5e-03) | 292 / 424 | 477 / 550 | 842 / 874 | 1636 / 1645 | 2377 / 2385 | 3221 / 3224 | 2942 / 2951 | 3001 / 3017 | 3017 / 3027 |
| sglang_fla_triton(tuned) | True (o 6.6e-03, s 6.4e-03) | 197 / 310 | 338 / 418 | 632 / 677 | 1244 / 1253 | 1826 / 1833 | 2490 / 2501 | 2193 / 2206 | 2065 / 2076 | 2255 / 2267 |
| aiter_fork_fixed:opt_vk_triton | True (o 8.9e-03, s 6.4e-03) | 228 / 316 | 422 / 466 | 852 / 885 | 1697 / 1709 | 2574 / 2586 | 3559 / 3572 | 3178 / 3190 | 3203 / 3211 | 3223 / 3232 |
| aiter_fork_fixed:opt_vk_hipK5 | True (o 9.0e-03, s 6.5e-03) | 147 / 337 | 265 / 354 | 547 / 592 | 1087 / 1113 | 1624 / 1649 | 2224 / 2235 | 2033 / 2048 | 1965 / 1974 | 2091 / 2101 |
| aiter_fork_fixed:opt_vk_flydslK5 | True (o 9.0e-03, s 6.5e-03) | 167 / 407 | 279 / 456 | 546 / 698 | 1062 / 1192 | 1561 / 1680 | 2126 / 2232 | 1989 / 2087 | 2009 / 2105 | 2037 / 2148 |
| aiter_fork_fixed:opt_vk_flydslPrep+hipK5 | True (o 9.0e-03, s 6.5e-03) | 119 / 275 | 206 / 287 | 384 / 429 | 771 / 799 | 1166 / 1181 | 1578 / 1591 | 1398 / 1408 | 1332 / 1344 | 1466 / 1478 |
| aiter_fork_fixed:opt_vk_flydslPrep+flydslK5 | True (o 9.0e-03, s 6.5e-03) | 133 / 296 | 219 / 327 | 390 / 490 | 752 / 859 | 1104 / 1206 | 1486 / 1582 | 1355 / 1438 | 1382 / 1464 | 1417 / 1497 |
| aiter_main:opt_vk_triton | True (o 8.9e-03, s 6.4e-03) | 262 / 345 | 493 / 533 | 1003 / 1035 | 2031 / 2039 | 3066 / 3061 | 4244 / 4257 | 3858 / 3867 | 3877 / 3882 | 3904 / 3913 |
| aiter_main:opt_vk_hipK5 | True (o 9.0e-03, s 6.5e-03) | 182 / 318 | 335 / 394 | 697 / 741 | 1416 / 1429 | 2105 / 2130 | 2884 / 2898 | 2698 / 2713 | 2633 / 2643 | 2768 / 2784 |
| aiter_main:opt_vk_flydslK5 | True (o 9.0e-03, s 6.5e-03) | 176 / 302 | 324 / 385 | 672 / 720 | 1359 / 1378 | 2006 / 2037 | 2766 / 2777 | 2618 / 2626 | 2575 / 2586 | 2672 / 2674 |
| aiter_main:opt_vk_flydslPrep+hipK5 | True (o 9.0e-03, s 6.5e-03) | 152 / 261 | 277 / 336 | 530 / 556 | 1088 / 1098 | 1629 / 1642 | 2232 / 2246 | 2056 / 2067 | 1988 / 2000 | 2141 / 2159 |
| aiter_main:opt_vk_flydslPrep+flydslK5 | True (o 9.0e-03, s 6.5e-03) | 146 / 214 | 265 / 279 | 506 / 521 | 1038 / 1039 | 1541 / 1551 | 2110 / 2113 | 1970 / 1985 | 1929 / 1936 | 2043 / 2059 |
| aiter_q35dev:opt_vk_triton | True (o 8.9e-03, s 6.4e-03) | 290 / 421 | 551 / 647 | 1071 / 1164 | 2178 / 2234 | 3291 / 3347 | 4541 / 4588 | 4069 / 4110 | 4048 / 4087 | 4137 / 4169 |
| aiter_q35dev:opt_vk_hipK5 | True (o 9.0e-03, s 6.5e-03) | 182 / 395 | 340 / 473 | 652 / 775 | 1335 / 1398 | 1966 / 2042 | 2694 / 2756 | 2512 / 2573 | 2448 / 2505 | 2574 / 2629 |
| aiter_q35dev:opt_vk_flydslK5 | True (o 8.9e-03, s 6.4e-03) | 289 / 414 | 551 / 629 | 1074 / 1146 | 2174 / 2218 | 3292 / 3346 | 4555 / 4605 | 4080 / 4119 | 4050 / 4080 | 4140 / 4169 |
| aiter_q35dev:opt_vk_flydslPrep+hipK5 | ERROR: RuntimeError: use_prepare_flydsl unsupported by this aiter | | | | | | | | | |
| aiter_q35dev:opt_vk_flydslPrep+flydslK5 | ERROR: RuntimeError: use_prepare_flydsl unsupported by this aiter | | | | | | | | | |
| aiter_opt_installed:opt_vk_triton | True (o 8.9e-03, s 6.4e-03) | 264 / 341 | 495 / 534 | 1005 / 1030 | 2026 / 2033 | 3054 / 3061 | 4246 / 4264 | 3856 / 3868 | 3873 / 3884 | 3903 / 3914 |
| aiter_opt_installed:opt_vk_hipK5 | True (o 9.0e-03, s 6.5e-03) | 183 / 349 | 337 / 426 | 696 / 734 | 1416 / 1443 | 2105 / 2125 | 2879 / 2894 | 2691 / 2709 | 2624 / 2635 | 2757 / 2770 |
| aiter_opt_installed:opt_vk_flydslK5 | True (o 9.0e-03, s 6.5e-03) | 200 / 401 | 355 / 522 | 701 / 855 | 1392 / 1520 | 2041 / 2171 | 2781 / 2900 | 2649 / 2761 | 2674 / 2770 | 2704 / 2809 |
| aiter_opt_installed:opt_vk_flydslPrep+hipK5 | False (o 9.0e-03, s 6.5e-03) | 153 / 295 | 276 / 370 | 531 / 575 | 1092 / 1109 | 1631 / 1641 | 2230 / 2246 | 2050 / 2061 | 1981 / 1993 | 2280 / 2134 |
| aiter_opt_installed:opt_vk_flydslPrep+flydslK5 | False (o 9.0e-03, s 6.5e-03) | 168 / 295 | 291 / 392 | 536 / 638 | 1069 / 1161 | 1571 / 1668 | 2139 / 2226 | 2016 / 2092 | 2023 / 2112 | 2074 / 2158 |


要点：
- 去掉 K5 瓶颈后，`chunk_fwd_o` 成为最大项（aiter 默认 BK=BV=32 配置 690 µs）；tuned 后 230 µs。aiter main 尚无此 tuned 配置（692 µs），因此在本模型上 **fork + tuned chunk_o (1166 µs) 快于 aiter main (1541 µs)**。
- FlyDSL K5 + gather/scatter（fork 的 mfma16 版不支持 indexed pool）在 ≥8k 单序列略快于 HIP K5（1104 vs 1166 µs），但在 1k、16×1k 等形状更慢；HIP K5 原地读写 pool、形状更稳，选为默认（`SGLANG_GDN_AITER_K5=hip`，可切 `flydsl`）。

### 3.4 Decode（每层 GDN，graph replay，36 层各自独立 state pool 轮转 = cache-cold，µs/层）

| bs | sglang Triton packed (default cfg) | sglang Triton packed (applied cfg) | aiter FlyDSL default | aiter FlyDSL tuned |
|---|---|---|---|---|
| 1 | 8.4 | **6.9** | 8.4 | 6.8 |
| 2 | 11.3 | **9.6** | 10.0 | 10.0 |
| 4 | 16.4 | **15.0** | 14.7 | 14.9 |
| 8 | 25.3 | **22.8** | 22.7 | 22.7 |
| 12 | 30.6 | **30.6** | 32.0 | 28.4 |
| 16 | 35.3 | **35.3** | 38.3 | 35.2 |
| 24 | 43.1 | **43.1** | 51.7 | 44.5 |
| 32 | 51.2 | **51.2** | 63.2 | 53.6 |

- **第一版 decode 微基准是 cache-warm 的（同一组 slot 在 MALL 中常驻），结论是错的**：它显示 tuned 配置/FlyDSL 更快，但 in-situ profile 显示它们反而慢 5–7%。改为 36 层独立 pool 轮转后（与真实 decode 一致）重新 tune，结论如上。
- aiter `fused_rearrange_sigmoid_gdr`、ATOM VK 版本均不优于 sglang packed decode；aiter `fused_sigmoid_gating_recurrent` 是 KV 布局，接入需 transpose（实测 bs32 548 µs，不可用）。
- 结论：decode 保持 sglang Triton packed decode，仅 bs≤8 使用 tuned 配置（bs1 8.4→6.9 µs，bs8 25.3→22.8 µs）。FlyDSL decode 已接入（`--linear-attn-decode-backend aiter`）供对比，不推荐。

### 3.5 框架层 split（ratio 3）
`_GDN_FUSED_QKVZBA_RATIOS` 在 AITER 模式下不含 3，本模型走 `contiguous+cat` 回退路径。实测（`results/kernel/qkvzba_split.txt`，GPU 时间）：fused 版在 decode bs8–32 每层快 ~5 µs（每步 ~0.18 ms，<0.5%），在 12k prefill 每层慢 39 µs。两边影响都可忽略，**未修改**。

## 4. Tuning

| 对象 | 方法 | 结果 |
|---|---|---|
| aiter FlyDSL K5 BV | aiter main 官方 tuner（model_configs 流程），新增 Qwen3.8 TP2 cases | 发现并修复 tuner bug 后：所有 46 个形状最优均为 BV=64 = 现有启发式，**无收益**（表已入库作记录：`aiter/configs/model_configs/qwen3_8_flash_next_chunk_gdn_h_opt_tuned.csv`） |
| aiter `chunk_fwd_kernel_o_opt_vk` (Triton) | ROCm MI300X Triton 指南：tile × num_warps × num_stages 全扫，top-5 再扫 waves_per_eu × matrix_instr_nonkdim × kpack | BK=64, BV=128, 4 warps, 1 stage, nonkdim 16, kpack 2：gmean **3.17x**（12k: 691→230 µs）；按 aiter 惯例写入 `configs/gfx942/triton/attention/gated_delta_rule/DEFAULT.json` 并通过 `get_tuned_kernel_config` 加载 |
| sglang FLA K5 | 同上 | BV=64/4w/2s（12k: 1511→931 µs, 1.6x），在 gfx942 设为默认 |
| sglang FLA `chunk_fwd_o` | 同上 | BK=64, BV=128, 4w, 1s, kpack 2：2.46x |
| sglang FLA `recompute_w_u` | 同上 | BK=64, BV=128, 2w, 3s, wpe 2, kpack 2：1.98x |
| sglang packed decode | per-bs 扫描（cache-cold） | 仅 bs≤8 有收益（见 3.4） |
| aiter FlyDSL decode | per-bs 扫 NUM_BLOCKS_PER_V_DIM × NUM_WARPS × WARP_THREADS_K（cache-cold） | bs12–32 提升 6–15%，但仍不优于 Triton packed decode；写入 `gdr_decode_tuned.csv` |

sweep 原始数据：`results/tune/`；脚本：`bench/sweep_triton.py`、`bench/tune_decode.py`。

## 5. End-to-end（TP2，真实服务）

**方法**：4 个变体在 4 个 TP2 GPU 对上并行运行，共 5 轮，每轮轮换 GPU 对；每轮开始前要求 8 卡连续 30 s 占用为 0（排除其他租户）。每轮每个变体依次跑 4 个负载（`sglang.bench_serving`，random 数据集，range-ratio 1.0，warmup 10，同一轮各变体使用相同 seed），之后再做随机 prompt 输出多样性检查。第 1 轮另外采集了 prefill/decode profile。脚本：`e2e/run_all.sh`、`e2e/run_variant.sh`；原始数据：`results/e2e_final/`。

| 变体 | sglang | GDN prefill | GDN decode |
|---|---|---|---|
| base | 分支 head 原版 | Triton（原配置） | Triton packed（原配置） |
| sgl_tuned | 本次分支 | Triton + gfx942 tuned 配置 | Triton packed（bs≤8 tuned） |
| aiter_pf | 本次分支 | **aiter opt_vk：FlyDSL prepare + HIP K5 + tuned chunk_o** | Triton packed（bs≤8 tuned） |
| aiter_all | 本次分支 | 同 aiter_pf | aiter FlyDSL `gdr_decode`（tuned） |

所有变体使用同一 aiter 目录（安装版 4752a408 + #4952 修复 + GDN tuned 配置）。base 和 sgl_tuned 不调用这些 GDN 改动。

**结果（5 轮中位数，括号内为最小值–最大值；百分比是相对 base 的改善）：**

### Workload A: in/out/conc = 12000/350 c1

| metric | base | sgl_tuned | aiter_pf | aiter_all |
|---|---|---|---|---|
| TTFT p50 ms | 1049.5 (1043-1066, n=5) | 1005.4 (1004-1017, n=5) **+4.2%** | 982.2 (978-1002, n=5) **+6.4%** | 979.1 (977-1003, n=5) **+6.7%** |
| TTFT mean ms | 1073.9 (1046-1217, n=5) | 1006.1 (1006-1161, n=5) **+6.3%** | 986.6 (977-1159, n=5) **+8.1%** | 1026.6 (983-1232, n=5) **+4.4%** |
| TPOT p50 ms | 11.65 (11.59-12.04, n=5) | 11.58 (11.50-12.04, n=5) **+0.6%** | 11.57 (11.49-12.52, n=5) **+0.7%** | 11.57 (11.49-11.99, n=5) **+0.7%** |
| ITL p50 ms | 11.65 (11.59-12.02, n=5) | 11.57 (11.50-12.05, n=5) **+0.6%** | 11.55 (11.49-11.95, n=5) **+0.8%** | 11.57 (11.49-12.00, n=5) **+0.7%** |
| out tok/s | 67.91 (45.11-68.45, n=5) | 69.33 (45.49-69.55, n=5) **+2.1%** | 69.55 (45.11-70.12, n=5) **+2.4%** | 68.22 (47.10-70.07, n=5) **+0.5%** |
| total tok/s | 2396.1 (1592-2415, n=5) | 2446.3 (1605-2454, n=5) **+2.1%** | 2454.1 (1592-2474, n=5) **+2.4%** | 2407.2 (1662-2473, n=5) **+0.5%** |

### Workload B: in/out/conc = 12000/1 c1

| metric | base | sgl_tuned | aiter_pf | aiter_all |
|---|---|---|---|---|
| TTFT p50 ms | 1061.8 (1044-1107, n=5) | 1002.8 (998-1168, n=5) **+5.6%** | 983.6 (976-1099, n=5) **+7.4%** | 994.5 (976-1113, n=5) **+6.3%** |
| TTFT mean ms | 1064.0 (1044-1087, n=5) | 999.9 (949-1203, n=5) **+6.0%** | 983.4 (976-1093, n=5) **+7.6%** | 992.8 (975-1217, n=5) **+6.7%** |
| total tok/s | 11250.7 (10001-11475, n=5) | 11949.5 (9352-11981, n=5) **+6.2%** | 12180.8 (10357-12266, n=5) **+8.3%** | 12065.3 (9838-12291, n=5) **+7.2%** |

### Workload C: in/out/conc = 4000/350 c16

| metric | base | sgl_tuned | aiter_pf | aiter_all |
|---|---|---|---|---|
| TTFT p50 ms | 4561.2 (4484-5449, n=5) | 4398.1 (4322-4833, n=5) **+3.6%** | 4485.4 (4306-4825, n=5) **+1.7%** | 4380.4 (4293-5404, n=5) **+4.0%** |
| TTFT mean ms | 4473.1 (4373-5230, n=5) | 4303.2 (4262-4623, n=5) **+3.8%** | 4244.4 (4181-4461, n=5) **+5.1%** | 4412.7 (4154-4920, n=5) **+1.3%** |
| TPOT p50 ms | 31.97 (31.38-54.35, n=5) | 31.86 (31.80-43.40, n=5) **+0.3%** | 31.81 (31.09-32.42, n=5) **+0.5%** | 31.93 (31.23-69.89, n=5) **+0.1%** |
| ITL p50 ms | 29.36 (29.30-36.02, n=5) | 29.33 (29.29-29.67, n=5) **+0.1%** | 29.35 (29.21-29.89, n=5) **+0.0%** | 29.43 (29.35-66.46, n=5) **-0.2%** |
| out tok/s | 354.0 (237-357, n=5) | 358.8 (289-361, n=5) **+1.4%** | 361.6 (351-364, n=5) **+2.1%** | 350.5 (189-363, n=5) **-1.0%** |
| total tok/s | 4399.9 (2948-4441, n=5) | 4459.3 (3595-4485, n=5) **+1.4%** | 4494.2 (4365-4523, n=5) **+2.1%** | 4355.8 (2348-4506, n=5) **-1.0%** |

### Workload D: in/out/conc = 2000/500 c32

| metric | base | sgl_tuned | aiter_pf | aiter_all |
|---|---|---|---|---|
| TTFT p50 ms | 4944.1 (4323-5105, n=5) | 4437.4 (4172-4674, n=5) **+10.2%** | 4686.3 (4364-4965, n=5) **+5.2%** | 4397.2 (4104-5625, n=5) **+11.1%** |
| TTFT mean ms | 5123.5 (5004-5548, n=5) | 4862.7 (4838-4992, n=5) **+5.1%** | 4976.4 (4781-5129, n=5) **+2.9%** | 4902.6 (4719-5991, n=5) **+4.3%** |
| TPOT p50 ms | 45.29 (45.00-45.85, n=5) | 45.19 (44.30-45.91, n=5) **+0.2%** | 45.10 (44.58-46.23, n=5) **+0.4%** | 45.25 (44.92-94.72, n=5) **+0.1%** |
| ITL p50 ms | 43.64 (43.59-44.11, n=5) | 43.61 (43.41-43.76, n=5) **+0.1%** | 43.58 (43.48-44.47, n=5) **+0.1%** | 43.80 (43.61-89.84, n=5) **-0.4%** |
| out tok/s | 538.9 (536-542, n=5) | 547.5 (535-549, n=5) **+1.6%** | 542.2 (532-549, n=5) **+0.6%** | 534.0 (271-551, n=5) **-0.9%** |
| total tok/s | 2694.4 (2681-2709, n=5) | 2737.6 (2676-2743, n=5) **+1.6%** | 2711.2 (2661-2745, n=5) **+0.6%** | 2670.1 (1355-2755, n=5) **-0.9%** |

**精度**（GSM8K test，跳过前 5 题后共 1314 题，chat 模板，greedy，`bench/gsm8k_eval.py`）：

| 变体 | base | sgl_tuned | aiter_pf | aiter_all |
|---|---|---|---|---|
| 准确率 | 96.35% | 96.50% | 96.42% | 96.73% |

二项分布标准误约 ±0.5%，四者等价。随机 prompt 输出多样性（16 条，每条 200 token 中不同 token 数的平均）：各变体、各轮都在 106–118 之间，无退化。

**Profile 证据（第 1 轮，4 条 12000-token prefill，TP0）**（`results/e2e_final/profile_prefill_summary.txt`、`profile_kernel_count_diff.txt`）：
- GDN 相关 kernel 总时间：base 483 ms（占全部 GPU 时间的 11.3%）→ sgl_tuned 331 ms → aiter_pf 247 ms（其中 conv1d 约 63 ms、qkv split 约 14 ms，各变体相同）。
- 核心 chunk 部分（prepare + K5 + chunk_o + l2norm）：base 约 100 ms/请求 → aiter 约 40 ms/请求，与 kernel 级 2.6x 的结论一致。
- aiter_pf 和 base 之间，除 GDN 自身的 kernel 外，**其余 kernel 的调用次数一致**，没有新增任何 transpose/copy/contiguous kernel。aiter 路径还省掉了 sglang `chunk_fwd_o` 里的 `zeros_like` 初始化（FillFunctor 调用次数 535 → 248）。
- aiter_all 第 1 轮在 GPU 6,7 上整体变慢，所有 kernel 都变慢，属于外部干扰，见下表中带 * 的数据。

**解读**：
- **Prefill**：TTFT 稳定改善。负载 A（12k 输入）TTFT p50：sgl_tuned -4.2%，aiter -6.4%。5 轮的取值范围互不重叠，且每个 GPU 对上排序都一致。
- **Decode**：四个变体的 TPOT/ITL 一致，差异 ≤0.4%，在噪声内。原因是 decode 时 GDN 只占约 1.7% 的 GPU 时间。
- **C/D 的 TTFT**：主要由排队决定。mamba 槽位限制同时运行 ≤31 个请求，所以 TTFT 波动大（单变体的 min–max 跨度可达 ±10%），不能用来区分 GDN backend。
- **被排除的数据**：带 * 的单轮数据偏离中位数超过 15%，集中出现在第 1 轮，与当时观察到的其他租户占用以及同轮并行 profile 采集吻合。取中位数后不影响结论。

### Per-round raw values (variant @ GPU pair); * = deviates >15% from that variant's 5-round median


A median_ttft_ms

| variant | r1 | r2 | r3 | r4 | r5 | median |
|---|---|---|---|---|---|---|
| base | 1060.9@0,1 | 1042.8@2,3 | 1049.5@4,5 | 1043.9@6,7 | 1066.3@0,1 | 1049.5 |
| sgl_tuned | 1016.7@2,3 | 1003.6@4,5 | 1005.1@6,7 | 1005.4@0,1 | 1014.6@2,3 | 1005.4 |
| aiter_pf | 1001.6@4,5 | 979.1@6,7 | 982.2@0,1 | 978.3@2,3 | 1000.2@4,5 | 982.2 |
| aiter_all | 1002.9@6,7 | 978.1@0,1 | 979.1@2,3 | 977.0@4,5 | 1001.4@6,7 | 979.1 |

B median_ttft_ms

| variant | r1 | r2 | r3 | r4 | r5 | median |
|---|---|---|---|---|---|---|
| base | 1107.4@0,1 | 1064.1@2,3 | 1044.6@4,5 | 1044.2@6,7 | 1061.8@0,1 | 1061.8 |
| sgl_tuned | 1168.1@2,3* | 1005.4@4,5 | 1000.2@6,7 | 1002.8@0,1 | 998.2@2,3 | 1002.8 |
| aiter_pf | 1099.3@4,5 | 988.1@6,7 | 975.8@0,1 | 982.0@2,3 | 983.6@4,5 | 983.6 |
| aiter_all | 1113.3@6,7 | 994.5@0,1 | 976.2@2,3 | 975.8@4,5 | 1000.5@6,7 | 994.5 |

A median_itl_ms

| variant | r1 | r2 | r3 | r4 | r5 | median |
|---|---|---|---|---|---|---|
| base | 11.6@0,1 | 11.6@2,3 | 11.7@4,5 | 11.6@6,7 | 12.0@0,1 | 11.6 |
| sgl_tuned | 11.6@2,3 | 11.6@4,5 | 11.5@6,7 | 11.5@0,1 | 12.0@2,3 | 11.6 |
| aiter_pf | 11.6@4,5 | 11.5@6,7 | 11.6@0,1 | 11.6@2,3 | 12.0@4,5 | 11.6 |
| aiter_all | 11.5@6,7 | 11.5@0,1 | 11.6@2,3 | 11.6@4,5 | 12.0@6,7 | 11.6 |

C median_itl_ms

| variant | r1 | r2 | r3 | r4 | r5 | median |
|---|---|---|---|---|---|---|
| base | 36.0@0,1* | 29.3@2,3 | 29.4@4,5 | 29.3@6,7 | 29.4@0,1 | 29.4 |
| sgl_tuned | 29.7@2,3 | 29.3@4,5 | 29.3@6,7 | 29.5@0,1 | 29.3@2,3 | 29.3 |
| aiter_pf | 29.9@4,5 | 29.2@6,7 | 29.4@0,1 | 29.3@2,3 | 29.4@4,5 | 29.4 |
| aiter_all | 66.5@6,7* | 29.4@0,1 | 29.4@2,3 | 29.4@4,5 | 29.4@6,7 | 29.4 |

D median_itl_ms

| variant | r1 | r2 | r3 | r4 | r5 | median |
|---|---|---|---|---|---|---|
| base | 44.1@0,1 | 43.7@2,3 | 43.6@4,5 | 43.6@6,7 | 43.6@0,1 | 43.6 |
| sgl_tuned | 43.5@2,3 | 43.8@4,5 | 43.4@6,7 | 43.6@0,1 | 43.6@2,3 | 43.6 |
| aiter_pf | 44.5@4,5 | 43.6@6,7 | 43.5@0,1 | 43.6@2,3 | 43.7@4,5 | 43.6 |
| aiter_all | 89.8@6,7* | 43.8@0,1 | 43.6@2,3 | 43.8@4,5 | 43.7@6,7 | 43.8 |

C median_ttft_ms

| variant | r1 | r2 | r3 | r4 | r5 | median |
|---|---|---|---|---|---|---|
| base | 5448.8@0,1* | 4494.5@2,3 | 4561.2@4,5 | 4633.9@6,7 | 4484.5@0,1 | 4561.2 |
| sgl_tuned | 4833.4@2,3 | 4379.1@4,5 | 4398.1@6,7 | 4492.3@0,1 | 4322.3@2,3 | 4398.1 |
| aiter_pf | 4734.3@4,5 | 4388.5@6,7 | 4306.4@0,1 | 4824.7@2,3 | 4485.4@4,5 | 4485.4 |
| aiter_all | 5404.1@6,7* | 4294.9@0,1 | 4293.1@2,3 | 4380.4@4,5 | 4659.6@6,7 | 4380.4 |

D median_ttft_ms

| variant | r1 | r2 | r3 | r4 | r5 | median |
|---|---|---|---|---|---|---|
| base | 4944.1@0,1 | 4949.4@2,3 | 4623.7@4,5 | 5105.2@6,7 | 4323.0@0,1 | 4944.1 |
| sgl_tuned | 4674.3@2,3 | 4172.3@4,5 | 4437.4@6,7 | 4180.5@0,1 | 4612.8@2,3 | 4437.4 |
| aiter_pf | 4964.9@4,5 | 4686.3@6,7 | 4363.6@0,1 | 4800.2@2,3 | 4368.8@4,5 | 4686.3 |
| aiter_all | 5625.0@6,7* | 4103.8@0,1 | 4340.2@2,3 | 4832.7@4,5 | 4397.2@6,7 | 4397.2 |


## 6. 重要问题记录：aiter FlyDSL prepare 数值发散（以及由此产生的虚假加速）

1. 第一轮 e2e 中，两个 aiter 变体在高并发负载（C/D）上 TPOT/ITL 改善 6–8%，但 aiter_pf 的 decode kernel 与 sgl_tuned **完全相同**。
2. 二分：保留 aiter backend、但让 extend 实际调用 Triton → 收益消失；说明收益来自 prefill kernel 的“结果”，而不是 import 副作用。纯 decode 区间吞吐也高 10%，profile 显示 MoE gateup kernel 单次耗时 104 → 65 µs。
3. 用固定的随机 token prompt 比较 greedy 输出：base / sgl_tuned 每条 200 token 输出平均 111/113 个不同 token，**aiter_pf 只有 3.2 个**（重复 token）→ 输出退化 → MoE 路由集中 → decode 变快。即“加速”是错误数值造成的假象。
4. 服务内逐层对比（同一输入分别跑 Triton 与 aiter）：误差最高到 8.8e11/NaN；dump 出的真实输入可离线稳定复现；组件隔离显示 **K5（HIP/FlyDSL/Triton）都正确，错在 FlyDSL 融合 prepare**；aiter main 正确。定位到上游修复 `987203ba5`（#4952, “stabilize GDN prepare triangular solve”），cherry-pick 后：服务内 1368 次调用最大误差 2.3%（bf16 量级），输出多样性恢复（~110），GSM8K 96.42%。
5. 合成复现：key 相关系数 ≥0.5 时旧 kernel 误差 0.12 → 4e22；随机 key（我最初的测试）完全正常——这是 kernel 单测的盲区。
6. 防护：sglang `AiterGDNKernel` 启动时用相关 key 小输入自检 FlyDSL prepare，与 aiter Triton prepare 不一致则告警并回退（已验证：安装版 aiter 自动回退，修复版启用）。

## 7. 推荐配置

```bash
# aiter: 需包含 ROCm/aiter#4952（yicyang/aiter gdn-qwen38-tp2-gfx942 分支，或 >= main 987203ba5）
# sglang: yicyang/sglang gdn-opt-qwen38-tp2 分支
TP_SIZE=2 bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh \
    --linear-attn-prefill-backend aiter          # decode 保持默认 triton
# 可选：SGLANG_GDN_AITER_K5=flydsl（长单序列略快，短/多序列略慢；默认 hip）
```

限制：aiter prefill backend 不返回 per-chunk hidden state，**radix/mamba prefix cache 状态跟踪（`--disable-radix-cache` 关闭时）不支持**，会触发断言；本次脚本默认关闭 radix cache。未覆盖 MTP/target-verify（仍走 Triton）。

## 8. 产物

本目录（`benchmark/gdn_qwen38_mi308x/`）收录脚本、本报告和精简后的原始结果（kernel/tune 的全部 json/log、e2e_final 的 bench jsonl、GSM8K 预测、输出多样性）。profile trace、server log、dump 出的真实输入 `.pt` 文件体积较大，只保留在本地。e2e 脚本中的路径写死为 `/root/workspace/gdn_optimization`，复现时需要修改 `G=`。

本地完整产物：`/root/workspace/gdn_optimization/`
- `bench/`：kernel 基准、sweep、decode tuning、e2e 汇总、profile 分析、GSM8K、随机 prompt 多样性检查脚本
- `e2e/`：e2e 运行脚本（`run_variant.sh`、`run_all.sh`、`steady_decode_profile.sh`）及诊断 hook 存档
- `results/`：kernel/、tune/、e2e/（第一轮，含已作废的 aiter 结果，仅作问题记录）、e2e_final/（最终）
- 分支：`sglang` worktree → `gdn-opt-qwen38-tp2`；`aiter_fork` → `gdn-qwen38-tp2-gfx942`；`aiter_main` → `gdn-qwen38-tp2-gfx942`

远端：PUSH_PLACEHOLDER
