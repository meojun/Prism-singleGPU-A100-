# Pre-overlap / conservative-tau diagnostic

The completed Arm C cases in this result tree predate the two-phase overlap
fix and use `tau=0.17108602804407128`. They are retained as diagnostic raw
evidence only and must not be reported as the final paper-faithful Arm C.

Arm A is independent of tau and the overlap implementation. Its completed raw
cases remain the final reusable released-prototype baseline. The resumable
supervisor was switched to `FC_RUN_C=0` after `C_bursty_r14_s1` completed, so
the remaining baseline cells run without spending GPU time on obsolete Arm C.
