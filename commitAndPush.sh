#!/usr/bin/env bash


if [ $# -eq 0 ]
  then
    comment="debugging"
  else
    comment="$@"
fi


git add commitAndPush.sh handlers.py main.py
git add README.md utils.py Cam.py Face_recognizer.py Journal.md
git commit -m "$comment";
git push origin master