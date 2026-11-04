# Source pins

Every Marin commit cited by a run record under `evidence/` that is not on Marin main. All of them are parents of this branch, so `git checkout <sha>` works.

| Commit | Date (UTC) | Subject |
|---|---|---|
| `f41b27c558222fac6404546280339ce37a0dfa72` | 2026-10-14 12:56 | [rolloutengine] Drop the VerifyIt re-export shim |
| `caba0fa38b780af42da54af2f188ba5bfcc28501` | 2026-10-23 03:12 | Backport #10351: defer ShellSim packages until tool execution |
| `f5ef16c8bc4030cffbb3c10636495c9fd6100bcf` | 2026-10-24 12:56 | Complete rank-one math comparison and experiment closeout |
| `9e3857f86b168605b063e4b8f5aedb34ad10f01c` | 2026-10-26 05:22 | m16 gate: add an 'opd' candidate kind bound by its export audit, readback and run id |
| `149c1dae21054fc5df01ae794d14717afd54a1cb` | 2026-10-28 03:11 | [post-training] Bind T3 sampling to the original native SFT loop |
| `4bc3742fe7726188ed001daecf805d97cd191a20` | 2026-10-29 22:20 | Separate special-token loss weights from frozen row membership |
| `dce28ebcd38e63f7c175b8769f8133fe6ffc08a9` | 2026-10-29 22:32 | Bind screen-owner trial types and scoped CLI interception |
| `3f8a7d7f47038604e3da7c47e4e3ec0fa5ea8bad` | 2026-10-29 23:03 | Bundle unseen inputs and fix Evalchemy keyword forwarding |
| `c1cea8dbf2016534b3b7984c42e1c44618bfbf5d` | 2026-10-30 00:27 | Bind screen children to parent candidates and preserve execution snapshots |
| `6ac2c53d061156ad22be53b332b7bd677a856290` | 2026-10-30 01:08 | Add five controlled BFCL SFT ablations |
| `278502ec5cee8b729b18bd0fa59b23cfa068e3fb` | 2026-10-30 02:27 | Add export-only recovery for the preempted ablation |
| `ac986b08026c07ab5096535361bccc36bf01d553` | 2026-10-30 03:32 | Load scoped screen owners directly from the committed workspace |
| `e007144ae77f5af957bafab798ffa87c04eced1f` | 2026-10-30 03:53 | Bind user-only teacher generation for the identity ablation |
| `818b2fe3b71d38251deb96e0895dbb8d7d338908` | 2026-10-30 04:04 | Bind four A5 capture shards to the exercised serving executor |
| `1e3294fd7ecd433bf9a0636cb0d60976de7d054e` | 2026-10-30 05:14 | Add a stratified identity-teacher pilot with explicit self-knowledge instructions |
| `ef8234140f041987d9d47cca668b8fba4fa4a0ec` | 2026-10-30 06:42 | Bind the existing A4 export to the scoped identity screen |
| `1e1b82d032b52926430e3d4fa377fffe0226e148` | 2026-10-30 07:42 | Bind native ON prefill for A3 identity evaluation |
| `cc2c0160d01bde4ef38f0eb92a0e6ce4c30b9dfb` | 2026-10-30 07:46 | Admit GLM ON with OFF fallback as A6 ablation |
| `638f597e73a0e47e9c3a2ddce5be8e6d8f5903af` | 2026-10-30 08:07 | Bind A6 to the original identity132 screen callback |
| `6be498f450c083fc784cc1222e775558bf2a23d3` | 2026-10-30 08:51 | Bind A3 native-prefill identity teacher pilot with closed-span QC |
| `a31fce76c9aea527c19a7c1dae6a02c79d8fb0c3` | 2026-10-30 09:02 | Scope A7 name affirmation and verify returned prompt IDs |
| `4bfe87adbb03372d9b51ff4c87b8f3a637e54bfe` | 2026-10-30 09:54 | Admit nested GLM ON-share ablation arms |
| `6f3d23355c7a5a44b0c50817fb537ffb1e0a7d2f` | 2026-10-30 10:03 | Bind the original identity screen to A8 dose arms |
| `1ffad15de6e32f75f12cdc9e283263dcc4cb7c42` | 2026-10-30 13:42 | Bind published v0.1 to the original identity screen |
| `5232f73084939ae2860cf4607151e8fa87aee156` | 2026-10-30 13:46 | Bind fresh paired maths to a math-only lifetime owner |
| `4f49e3b016fb8731a19a3e9b3e2d06d65b7a9370` | 2026-10-30 13:56 | Place paired maths CPU owners explicitly in RNO |
| `8c44a21a1aae7afb67277b1b048f92b57f88cb20` | 2026-10-30 14:01 | Extend the frozen A8a recipe to 72 updates |
| `cf4da6f40326df150eac712e2eb323667f71a267` | 2026-10-30 14:45 | Admit the scoped TULU-style identity ablation arm |
| `43bba4a16d175072b20fea27b8fb5524c377c79f` | 2026-10-30 14:45 | Bind regional native agentic evaluation lifetime policy |
| `62550b5c905e63e57121576edf8806419e76d0bb` | 2026-10-30 15:14 | Capture original identity screens with native token boundaries |
| `fd227ed536067409eb2532f050027e100ac99bd4` | 2026-10-30 15:39 | Bind future identity screens to actual audited export readiness |
| `36627ae6560aab1e1dc75adc26970436ba23f6eb` | 2026-10-30 16:11 | Bind one export recovery and one fresh retrain after storage failure |
| `0c15bfbe2918c88e60c76371dee04c7bff52a449` | 2026-10-30 16:12 | Bind native identity screens to the fresh recovery exports |
| `04459e4e966308f0f35fb8aff62c08071bfee17a` | 2026-10-30 17:03 | Bind A3 paired maths to the existing evaluation method |
| `741e5928030cdc0e2a4b0f0199ad3caf881e60c3` | 2026-10-30 17:10 | Add the 72-update identity OFF ablation arm |
| `a631378869c48f969f11a41b13a525c517843702` | 2026-10-30 17:15 | Archive protected Path B checkpoints to private Hub repositories |
| `4b810001e66d339031b53cd0057b3904c2354a02` | 2026-10-30 17:22 | Admit the A11 label at both native screen CLI boundaries |
| `d3461ff0a491c5cd0ba9a907e6d0870cc86ad2f7` | 2026-10-30 17:28 | Read private archive credentials from isolated secret files |
| `b99e099b441157c1f5836758521a21b5569f6531` | 2026-10-30 17:42 | Bind the one A3 maths infrastructure replacement candidate |
| `819f817ebbdb325b3bf8058623f3c254dc7eaadf` | 2026-10-30 18:58 | Bind A11 candidate for paired mathematics |
| `69ddc88fb2ff4a6e45ee5cbbec866d9f86c2d068` | 2026-10-30 23:06 | Add short-dose and doubled tool-replay ablation arms |
| `874ce94bad2fbb5478dc723f9edcdc63e56196aa` | 2026-10-30 23:36 | Bind audited CPU Path B linear merge sweep |
| `ef9d5ed97f4f24f1e133a8903a42e222c1831ace` | 2026-10-30 23:39 | Report bounded parent delta norms for merge readback |
| `462bba9353b562f841d095a6265dac47d5cafafe` | 2026-10-30 23:56 | Use virtual-host S3 transport for CPU merge workers |
| `8125c4faa1e7bb4ed9fa35e5bb9aaf45a3602362` | 2026-10-31 00:09 | Bind production export-only recovery for completed A12 checkpoint |
| `076b09efd89ddb1981c245a5cb3a8d2dcc3663a3` | 2026-10-31 01:09 | Bind native identity screen to admitted M51 merge provenance |
| `2989a8d6f1bc43ea6a43356a3e667172c59e2844` | 2026-10-31 01:09 | Bind production paired mathematics to admitted M51 merge |
| `3e1442d4f0ec2edce23804c0a7714d9249564966` | 2026-10-31 01:17 | Bind production native identity screen to recovered A12 export |
| `124a6ac6940d08f672e585b647c055c7b3be4894` | 2026-10-31 01:17 | Bind recovered original A12 model for paired mathematics |
| `7830afc88b2c0690c3876d13278b3c1899460bc8` | 2026-10-31 01:35 | Bind remaining native merge screens to actual admissions |
| `35a90ade762280a23d008487d5fc43e08bd30f72` | 2026-10-31 01:36 | Bind paired mathematics to fresh A13 and remaining merge exports |
| `0c9445ab1eae4f712ee3d264a42277a87bc3ce68` | 2026-10-31 01:38 | Record actual A13 export driver provenance |
| `cd8348baef2419b8a8ef305fa8308933690393a3` | 2026-10-31 06:26 | Archive the Path B A13 lead checkpoint to a private Hub repository |
