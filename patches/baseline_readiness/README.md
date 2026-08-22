# Baseline-readiness source snapshot

`prism-research/` is intentionally gitignored by the experiment repository and
is recreated by `bootstrap.sh`. The complete source working-tree delta used for
the Algorithm 2 validation and D2 migration-bug reproduction is therefore
stored in `prism_research_worktree.patch`.

Base source repository commit:

```text
595ec1f170e75a43897a7a2ad58ac5a9820aa2e8
```

Apply from a clean checkout of that commit:

```bash
git apply /path/to/Prism/patches/baseline_readiness/prism_research_worktree.patch
```

The patch contains all 21 modified and 11 untracked source files from the
diagnostic checkout, including the Algorithm 2 runtime-order integration and
the timestamp instrumentation used to reproduce the migration residency bug.
It intentionally preserves the unfixed `v4_resident` source-release behavior
documented in `exp/results/baseline-readiness/MIGRATION_DIAGNOSIS.md`.
