#!/usr/bin/env bash
# fa_launch_safe.sh -- launch an FA sim so it SURVIVES the agent/session dying.
#
# WHY THIS EXISTS: on 2026-07-27 eight runs (~10h wall-clock) were lost when an API
# session limit killed the agents. The launchers used `nohup ... &`, which only blocks
# SIGHUP -- the harness kills the whole PROCESS GROUP, which nohup does not protect
# against. `setsid` puts the sim in a NEW SESSION with no controlling terminal, so a
# group/session kill aimed at the agent cannot reach it.
#
# Usage:  fa_launch_safe.sh <tag> [seed] [timeout_cycles]
#   NOTE: TIMEOUT_CYCLES=N yields only N/2 CYCLES ($finish at N*1000+500ps, clk 2000ps).
#         Default 1600000 -> 800,000 cycles. Verify this exceeds the expected kernel length.
#
# Guarantees:
#   * sim reparented to init (PPID 1) -- VERIFIED, not assumed; exits nonzero if not
#   * stdout/stderr to /tmp/yruns/<tag>.out, stdin from /dev/null
#   * /tmp/yruns/<tag>.meta records LAUNCHED/DONE + exit status, so a finished run is
#     readable from disk WITHOUT the launching agent's context surviving
set -uo pipefail

TAG="${1:?usage: fa_launch_safe.sh <tag> [seed] [timeout_cycles]}"
SEED="${2:-12345}"
TMO="${3:-1600000}"
KDIR=/scratch/yrh/ai-workspace/kernel-gen/radiance-kernels/kernels/flash_attention_mx
RUNS=/tmp/yruns
mkdir -p "$RUNS"

OUT="$RUNS/$TAG.out"
META="$RUNS/$TAG.meta"

if [ -s "$OUT" ] && grep -q DONE "$META" 2>/dev/null; then
  echo "REFUSING: $TAG already has a completed run ($OUT). Move it aside first." >&2
  exit 2
fi

: > "$OUT"
printf 'TAG=%s SEED=%s TMO=%s (=%s cycles) LAUNCHED=%s\n' \
       "$TAG" "$SEED" "$TMO" "$((TMO/2))" "$(date '+%F %T')" > "$META"

# ---- RECORD THE CONFIGURATION, or the scored result is unattributable. ----
# A .score file that says "37.21%, 12/12" and cannot be tied to a flag set is nearly useless:
# it happened, and the run could not be credited to any lever without the launching agent's
# context still being alive. That defeats the point of on-disk results. Record, in order of
# reliability: the -D list from the build log, the relevant env, and a .text hash for provenance
# (the build is NOT byte-reproducible -- ~776 tail-metadata bytes differ -- so .text is the only
# sound identity).
{
  BLOG="/tmp/fa_build_$TAG.log"
  if [ -r "$BLOG" ]; then
    printf 'DEFINES %s\n' "$(grep -ohE '[-]D[A-Za-z_][A-Za-z0-9_]*(=[^ ]*)?' "$BLOG" | sort -u | tr '\n' ' ')"
  else
    printf 'DEFINES (no build log at %s)\n' "$BLOG"
  fi
  printf 'ENV FA_DEFS=%s FA_HOST_DEFS=%s\n' "${FA_DEFS:-}" "${FA_HOST_DEFS:-}"
  # Hash the RV32 GPU SEGMENTS, not `.text`.
  # `/tmp/flash_<tag>.elf` is the FUSED soc image: `.text` is the RV64 HOST and the GPU code
  # lives in `.rv32.seg0..seg4`. A `.text` hash is therefore VACUOUS for attributing a
  # GPU-side build -- verified: flash_ypk.elf and flash_vrR1.elf (wildly different flag sets)
  # share .text 07daffadb184d09f and differ only in rv32. A check that passes every pair of
  # builds can never fail, which is worse than no check.
  for E in "/tmp/flash_$TAG.elf" "/tmp/rad_$TAG.elf" "$KDIR/kernel.soc.elf"; do
    if [ -r "$E" ]; then
      printf 'ELF %s\nRV32_SHA %s\nHOST_TEXT_SHA %s\n' "$E" \
        "$(readelf -x .rv32.seg0 -x .rv32.seg1 -x .rv32.seg2 -x .rv32.seg3 -x .rv32.seg4 \
             "$E" 2>/dev/null | sha256sum | cut -c1-16)" \
        "$(readelf -x .text "$E" 2>/dev/null | sha256sum | cut -c1-16)"
      break
    fi
  done
} >> "$META" 2>/dev/null

# setsid => new session, immune to the agent's process-group teardown.
setsid nohup bash -c '
  cd "$1" || exit 97
  bash fa_run.sh "$2" "$2" "$3" "$4"
  printf "DONE rc=%s at %s\n" "$?" "$(date "+%F %T")" >> "$5"
' _ "$KDIR" "$TAG" "$SEED" "$TMO" "$META" < /dev/null >> "$OUT" 2>&1 &

LPID=$!
disown "$LPID" 2>/dev/null || true

# VERIFY detachment by SESSION ID, not by PPID.
# PPID==1 is the WRONG assertion for a chained launch: simv ends up a GRANDCHILD of the
# detached process, so its parent is legitimately an intermediate shell. An earlier PPID
# test reported "NOT DETACHED" for four runs that were perfectly healthy. What actually
# matters is that the job is in a DIFFERENT SESSION from this caller, since that is what a
# process-group/session teardown aimed at the agent cannot cross.
MY_SID=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')
JOB_SID=""; PP=""
for _ in $(seq 1 25); do
  sleep 0.2
  JOB_SID=$(ps -o sid= -p "$LPID" 2>/dev/null | tr -d ' ')
  PP=$(ps -o ppid= -p "$LPID" 2>/dev/null | tr -d ' ')
  [ -n "$JOB_SID" ] && break
  [ -z "$PP" ] && break     # already exec'd past this pid; check the session below
done

# If the launcher pid is gone, look for any process in a session that isn't ours.
DETACHED=no
if [ -n "$JOB_SID" ] && [ "$JOB_SID" != "$MY_SID" ]; then
  DETACHED=yes
elif [ -z "$JOB_SID" ] && pgrep -s "$LPID" >/dev/null 2>&1; then
  DETACHED=yes            # $LPID became a session leader; its children live there
fi
printf 'PID=%s PPID=%s SID=%s CALLER_SID=%s DETACHED=%s\n' \
       "$LPID" "${PP:-gone}" "${JOB_SID:-gone}" "$MY_SID" "$DETACHED" >> "$META"

if [ "$DETACHED" != yes ]; then
  echo "WARNING: $TAG may still be parented to this shell (PPID=${PP:-?}) -- it can die with the agent." >&2
  exit 1
fi
echo "LAUNCHED $TAG pid=$LPID -> $OUT (meta: $META)"
