#!/bin/bash
# usage: asmstats.sh file.o  -> one line of instruction statistics
D=$(aarch64-linux-gnu-objdump -d --no-show-raw-insn "$1" | grep -P '^\s+[0-9a-f]+:\t')
c() { grep -cP "$1" <<<"$D"; }
printf "total=%-7s mov=%-6s [sp]=%-6s vecfp=%-6s ext=%-5s trn=%-5s uzp=%-5s zip=%-5s rev64=%-5s tbl=%-4s ins=%-5s\n" \
  "$(wc -l <<<"$D")" "$(c '\t(mov|fmov)\s')" "$(c '\[sp')" "$(c '\tf(add|sub|mul|mla|mls|neg|div|max|min|abs)\s+v\d+\.(4s|2d)')" \
  "$(c '\text\s')" "$(c '\ttrn[12]\s')" "$(c '\tuzp[12]\s')" "$(c '\tzip[12]\s')" "$(c '\trev64\s')" "$(c '\ttbl\s')" "$(c '\tins\s')"
