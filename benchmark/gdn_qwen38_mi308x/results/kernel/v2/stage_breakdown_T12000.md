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
