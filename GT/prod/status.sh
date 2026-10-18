#!/bin/bash
# Summary of the production run: condor queue, finished/failed jobs per tag, items on disk.
W=/shared/rsaas/joeyw3/pvideo/_work
ssh -o BatchMode=yes vision-h01.cs.illinois.edu 'condor_q joeyw3 -af ClusterId JobStatus Args RemoteHost HoldReason' 2>/dev/null \
  | awk '{s[$2]++; c[$1" "$2]++} END {for (k in c) print "cluster/status", k, c[k]; print "status: 1=idle 2=run 5=held"}' | sort
for f in $W/logs/*.out; do t=$(basename $f | cut -d_ -f1); echo "$t $(grep -cE 'SELECT DONE|RENDER DONE|ASSIGN' $f) $(grep -cE '^exit [1-9]|Traceback' $f)"; done \
  | awk '{n[$1]++; ok[$1]+=($2>0); bad[$1]+=($3>0)} END {for (t in n) print "logs", t, "files", n[t], "done", ok[t], "with_errors", bad[t]}' | sort
for T in T1 T3 T4 T5 T6; do echo "$T items: $(ls /shared/rsaas/joeyw3/pvideo/$T/*.json 2>/dev/null | wc -l)  cands: $(cat $W/$T/cands/*.json 2>/dev/null | grep -c '"score"') in $(ls $W/$T/cands 2>/dev/null | wc -l) videos"; done
