#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
#
# Identify repositories that are forks.

import pandas as pd
import time
from github import Github # pip install PyGithub
from github.GithubException import RateLimitExceededException, UnknownObjectException



# load list of repos
df = pd.read_csv('../data/search_github_allcontribs_results__sizeL_sizeU_url.txt',
                 names=['sizeL', 'sizeU', 'url'])
df['user/repo'] = df['url'].apply(lambda s: "/".join(s.split(".com/", 1)[1].split("/")[0:2]))  # convenient


# create GitHub instance
# "secret.txt" is a github token
# https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/creating-a-personal-access-token
# give token repo/public_repo permission
ACCESS_TOKEN = open('secret.txt').read().strip()
g = Github(ACCESS_TOKEN)

# iterate and flag forks
fout_result = open("../data/check_fork_results__fork_parent.txt", 'w')
for row in df.iterrows():
    while True:
        try:
            repo = g.get_repo(row[1]['user/repo'])
            if repo.fork:
                print("[Fork]",  row[1]['user/repo'], flush=True)
                fout_result.write(row[1]['user/repo'] + "\t" + repo.parent.full_name + '\n')
                fout_result.flush()
            else:
                print("[Not fork]", row[1]['user/repo'], flush=True)
            break
        except RateLimitExceededException:
            print("[RateLimitExceededException]. Sleeping for 60 seconds.", flush=True)
            time.sleep(60)
            continue
        except UnknownObjectException:
            print("[UnknownObjectException]. Could not find", row[1]['user/repo'] + ".", flush=True)
            break

# exit
fout_result.close()
