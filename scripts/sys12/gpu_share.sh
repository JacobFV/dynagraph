#!/usr/bin/env bash
# GPU turn-taking between the INKLING sys12 pipeline and other agents' gpu_heavy jobs (flock is not FIFO).
# While another agent's job WAITS on ~/dynagraph/.gpu_heavy.lock, set ~/dynagraph/HOLD_SYS12: the sys12 pipeline
# pauses at its next step boundary, so the waiting job gets the lock when the current sys12 step ends. As soon as a
# non-sys12 job HOLDS the lock, clear the HOLD (the next sys12 step then queues behind it). Never preempts anything.
# Cheap bash loop (no compute). Stop: rm ~/dynagraph/.sys12_share_on
L=$HOME/dynagraph/.gpu_heavy.lock; H=$HOME/dynagraph/HOLD_SYS12; ON=$HOME/dynagraph/.sys12_share_on
touch $ON
while [ -f $ON ]; do
  waiting=0; running=0
  for p in $(pgrep -x flock); do
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$cmd" in *"$L"*) ;; *) continue;; esac
    case "$cmd" in *sys12*) continue;; esac
    if pgrep -P $p > /dev/null; then running=1; else waiting=1; fi
  done
  if [ $running = 1 ]; then rm -f $H
  elif [ $waiting = 1 ]; then [ -f $H ] || { touch $H; echo "[$(date -u +%FT%TZ)] HOLD set (other agent waiting)"; }
  else rm -f $H; fi
  sleep 5
done
rm -f $H
