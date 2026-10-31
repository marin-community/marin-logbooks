# KEEP — Path B A13 update36

Owner-selected retained lead (A13 replaced A11 as the lead on 2026-10-31 ~06:30Z); not a release acceptance.

Source: `s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-a13-preemption-replacement1-20261030-r1/A13/run/hf/updates-36`

Permanent same-bucket destination: `s3://marin-rno2a/marin/checkpoints/karan/snowball-identity-pathb/A13-u36`

Recipe: A8a recipe from fresh ADC0078a v0.1, seed310, 36 updates, batch 8, sequence 8192, lr 2e-6, warmup 2; 104 full-field freezes (audit: 104 frozen native / 398 trained master-cast tensors) and special-token rows frozen (token_embed `6e35ddd6631a213297ee90e8e3ad8f8ff6cfd4d16e077f9276aadb996ee643dc`, output_proj `6a4d4c30a4fc840618b4e80d47434aab1ea50f95a0006f8223496b9da1ff7783`).

Packed input SHA256: `f860030d1266c90d146ef07995a95ce7501989727cf226217de82a87408c15ba`. Source event SHA256 `998b91183e8fb3ad70c5502015ff0765c50b82d40f02424a77f7ec2a7fa456c9`; source 502-tensor audit SHA256 `4bcfe2c44bad2214129d27325e7dfcc427f63b8d86553a643bc92e81809202f2`.

BFCL 5 valid logical chains [40, 39, 34, 32, 29] of 123 in slot order, mean 34.8/123 = 28.2927% (slot 4 is a whole R2 after an infrastructure-incomplete original and R1; index `50cc4a7810e1fb0739af02b3c81a2f3be05a7953e4b235565f70ad4b4027973f`). Native role-aware original132 (marten ledger `7f1833032212b3468946127024d4f3e838d6e6ff80ccf799f898cb85f6ed4c58`): OFF name29/creator31/foreign1/format0, ON name35/creator32/foreign0/format1 of 52. Recipe delta vs A8a: every thinking-ON replay row in the tool families (tool_first_call, tool_call_after_result, tool_reply) duplicated once in place, then packed to 36 updates. identity-final124846/reasoning5388/replay-final612974.

Evidence: local notes `path-b/A13-BFCL-production-five-r1/`, `path-b/native-boundary-A13-r1/`, `path-b/ablation-a12-a13-r1/`; source training commit `69ddc88fb2ff`; full event and audit remain byte-equal in this directory.

No source deletion, native checkpoint lease/read, optimizer restore or re-export. All copying/large SHA streaming takes place on cw-rno2a CPU, never through the host or another region.


Verified preservation manifest SHA256: `e03dd64d524b3b67aabf801ed62f81e1425e3722d33ca583c74167795989ab27`. Every indexed shard and all source HF objects were full-stream SHA256 checked against the source. Source 502-tensor audit was copied byte-for-byte; this is not a new tensor audit or a quality/release PASS.
