#!/usr/bin/env bash


if [ $# -eq 0 ]
  then
    comment="bug fixed"
  else
    comment="$@"
fi


git add commitAndPush.sh handlers.py main.py README.md ids utils.py Cam.py
git commit -m "$comment";
git push origin master