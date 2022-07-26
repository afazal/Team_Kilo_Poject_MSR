#!/bin/bash -e
# 
# Loop over top repos and clone them. 

BASE_LOC=/storage/MSR2021/git_folders/
REPO_LIST=../data/repo_shortlist.txt
cd $BASE_LOC
while IFS= read -r line; do
  USER_NAME=$(echo $line | awk -F'/' '{print $1}');
  REPO_NAME=$(echo $line | awk -F'/' '{print $2}');
  git clone "git@github.com:"$USER_NAME"/"$REPO_NAME".git"
done < $REPO_LIST
