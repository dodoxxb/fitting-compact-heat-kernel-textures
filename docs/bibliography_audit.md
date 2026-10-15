# 参考文献核验与本轮改稿记录

更新核验日期：2026-10-15。32 条参考文献，其中 30 篇论文取得 Crossref 或 DataCite API 的匹配元数据，Cyr et al. 使用 PMLR 官方会议条目，RTXNTC 使用官方软件仓库。原有25条 DOI 核验保留；本轮新增5条 DOI 与1条 PMLR 信息。

## 出版版本与元数据修正

- HKTex 更新为 ECCV 2026 正式条目，306–323 页。
- Taming 3DGS 更新为 SIGGRAPH Asia 2024；作者顺序采用正式 DOI 和作者项目页 BibTeX 的顺序。
- SizeGS 更新为 ACM Multimedia 2025，8214–8223 页；不再将 2024 预印本年份作为会议出版年份。
- Objaverse 的 Crossref 作者字段误写 Ehsanit；与 arXiv 作者记录交叉核对后使用 Kiana Ehsani。
- Hypernetworks 论文的 DataCite 姓名顺序与论文首页不同；采用 Laurent Belcour，保留已核实的 arXiv DOI。
- KiloNeRF 的 Crossref 和 CVF 页码不同，本稿不填写有歧义的页码。
- DGM auditing：主页列为 PRICAI 2026，但本次未查得精确匹配的正式出版 DOI，因此引用已核实的 arXiv 版本，未虚构会议页码。

## 新增文献的用途

- DGM auditing：受控协议与公平归因，直接服务于本文的等更新对照。
- Continuous Language Diffusion as a Decoder-Interface Problem：用作表面误差与渲染误差差异的跨领域背景，不将语言实验作为纹理证据。
- Thread-Efficient Decoding for Neural Texture Compression：区分 GPU 执行效率与表示容量。
- Neural Texture Compression using Hypernetworks：区分逐材质拟合成本与已拟合纹理的解码成本。

## DOI 核验明细

| Key | DOI / 官方链接 | 版本年份 |
|---|---|---|
| fista | [10.1137/080716542](https://doi.org/10.1137/080716542) | 2009 |
| mfista | [10.1109/TIP.2009.2028250](https://doi.org/10.1109/TIP.2009.2028250) | 2009 |
| hyperntc | [10.48550/arXiv.2606.26913](https://doi.org/10.48550/arXiv.2606.26913) | 2026 |
| objaverse | [10.1109/CVPR52729.2023.01263](https://doi.org/10.1109/CVPR52729.2023.01263) | 2023 |
| gode | [10.48550/arXiv.2501.13558](https://doi.org/10.48550/arXiv.2501.13558) | 2025 |
| gso | [10.1109/ICRA46639.2022.9811809](https://doi.org/10.1109/ICRA46639.2022.9811809) | 2022 |
| dgm | [10.48550/arXiv.2605.29004](https://doi.org/10.48550/arXiv.2605.29004) | 2026 |
| decoderinterface | [10.48550/arXiv.2606.08810](https://doi.org/10.48550/arXiv.2606.08810) | 2026 |
| hktex | [10.1007/978-3-032-37595-7_17](https://doi.org/10.1007/978-3-032-37595-7_17) | 2026 |
| golubpereyra | [10.1137/0710036](https://doi.org/10.1137/0710036) | 1973 |
| localhk | [10.48550/arXiv.2609.14105](https://doi.org/10.48550/arXiv.2609.14105) | 2026 |
| taming3dgs | [10.1145/3680528.3687694](https://doi.org/10.1145/3680528.3687694) | 2024 |
| threadntc | [10.48550/arXiv.2608.27888](https://doi.org/10.48550/arXiv.2608.27888) | 2026 |
| rtxntc | [官方仓库](https://github.com/NVIDIA-RTX/RTXNTC) | 2026 年访问 |
| kilonerf | [10.1109/ICCV48922.2021.01407](https://doi.org/10.1109/ICCV48922.2021.01407) | 2021 |
| moe | [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538) | 2017 |
| audiocache | [10.1109/ICASSP49660.2025.10889043](https://doi.org/10.1109/ICASSP49660.2025.10889043) | 2025 |
| ntc | [10.1145/3592407](https://doi.org/10.1145/3592407) | 2023 |
| texf | [10.48550/arXiv.2609.23606](https://doi.org/10.48550/arXiv.2609.23606) | 2026 |
| sizegs | [10.1145/3746027.3755370](https://doi.org/10.1145/3746027.3755370) | 2025 |
| dbc | [10.2312/sr.20251199](https://doi.org/10.2312/sr.20251199) | 2025 |

核验时使用的 DOI API 原始响应不随发行包提供，本文件只保留核验结论与差异说明；核验不等于对引用论文所有科学结论作担保。

本轮未改变实验标量记录，也没有新增 GPU 实验。

## Restored scientific supplement references

Five citations supporting the restored proofs, quantization and controllable mixtures were checked through DOI APIs: CCCP (2003), AdaRound (2020), GPTQ (arXiv 2022), conditional image compression (arXiv 2019), and controllable residual restoration (arXiv 2019). The cited preprint years are not substituted with later conference years. All five API responses returned status 200 and matched titles and author identities. The current manuscript has 26 references: 25 DOI-verified papers and one official software reference.

## 15 October mechanism extension

Added Cohen–Migliorati2017, Cyr et al.2020, Neural Harmonic Textures,
Mip-Splatting, Frozen-PINN, and shared-Gaussian mipmapped SVBRDF compression.
New API responses and author/title records are in mechanism_bibliography_checks.json.
Cyr has no invented DOI. Frozen-PINN is cited as ICLR2026, with its arXiv DOI
(first submitted2024; v3 revised15April2026). Mip-Splatting is cited as CVPR2024;
its formal DOI10.1109/CVPR52733.2024.01839 and pages19447–19456 were also checked with Crossref, and the title link now uses the conference DOI. Community summaries were not
used to infer bibliographic facts or experimental effectiveness.
