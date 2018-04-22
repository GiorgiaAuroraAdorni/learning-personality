#!/bin/bash

sort "$1" | uniq -c | sort -n -r | sed -r 's/\s*(.*?)\s*$/\1/' | cat > "$2"