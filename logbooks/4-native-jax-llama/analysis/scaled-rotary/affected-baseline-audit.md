# Scaled-RoPE affected-test baseline audit

All13 failed tests reproduce on unchanged PR base `7b1aab6e2e6837cac10595dd54d1b4d60b7d6361` with matching test locations, nested traceback frames, `ModuleNotFoundError: No module named 'cloud'`, and subprocess exit1. The82 inspected test/launcher/vendor/runtime/config source objects are identical between that base and feature `598e1968819d3a2a85778841e4a3695ab6d89d8b`. No feature source fix is indicated by these failures.

Only4 of the13 nodes exist on local main `8428b67fba80a1f8bf9dfeabc552f411cbb82be0`; all4 also fail with the same nested import error. Main uses an older launcher and outer test line, so this is the same underlying error rather than an identical full traceback. The9 absent nodes were not copied onto main.

The replay used fresh isolated local clones, their own library source paths, the same existing shared UV environment, no-sync/offline settings and the complete original safety marker expression. It reran only13 base nodes and4 existing main nodes. No dependencies were installed, environments synced, locks edited or accelerator/remote jobs touched. Feature and baseline source trees remain clean.

The original affected run remains **1828passed,197skipped,13failed**. This audit establishes that these13 failures predate the feature in the inherited shared-environment setup; it does not claim a green suite or fix the nested SkyRL environment.

Evidence:

- Exact argv/environment: `/private/tmp/native-jax-scaled-rotary-pr-base-baseline-command.json` and `/private/tmp/native-jax-scaled-rotary-main-baseline-command.json`.
- Replay logs: `/private/tmp/native-jax-scaled-rotary-pr-base-baseline.log` and `/private/tmp/native-jax-scaled-rotary-main-baseline.log`.
- Structured per-node failures: matching `*-baseline.xml` files; fingerprints and complete failures in `/private/tmp/native-jax-scaled-rotary-affected-baseline-audit.json`.
- Source identities: `/private/tmp/native-jax-scaled-rotary-baseline-source-manifest.json`.
