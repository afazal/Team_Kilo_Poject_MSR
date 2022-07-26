#!/usr/bin/env python
# coding: utf-8
# In[3]:

get_ipython().system('pip install pipreqs')

# In[5]:

#current dir
cwd = os.getcwd()

# In[6]:

print(cwd)

# In[2]:

get_ipython().system('pip install PyGithub')

# In[1]:

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# search-github-allcontribs.py

import time
from github import Github # pip install PyGithub
from github.GithubException import RateLimitExceededException


# In[2]:


def exhaust_list(paginated_list, sizeL, sizeU, fout_result):
    """Get all results from a paginated list and sleep if rate limit is hit."""
    for file in paginated_list:
        while True:
            try:
                url = str(file.download_url)  # sends an API query
                break
            except RateLimitExceededException:
                print("RateLimitExceededException]. Sleeping for 60 seconds.")
                time.sleep(60)
                continue
        fout_result.write(f'{sizeL}, {sizeU}, {url}\n')
        fout_result.flush()


def post_query(query, msg=None):
    """Send a query and sleep if rate limit is hit."""
    while True:
        try:
            result = g.search_code(query, order='desc')
            num_results = int(result.totalCount)  # avoids further API calls
            break
        except RateLimitExceededException:
            print("[RateLimitExceededException] Sleeping for 60 seconds.")
            time.sleep(60)
            continue
    return result, num_results


# ====================
# sweep parameter
# ====================
size = 0  # size window in bytes
step_size = 500  # range of sizes considered (initially 500 bytes)

multiplier = 2.0  # how large is the resizing when we don't get enough results
min_result = 50  # how few results are too few?
probe_trigger = 20  # num of queries before we check if there's larger files left

# ===================
# initialization
# ====================

# Create GitHub instance
# "secret.txt" is a github token
# https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/creating-a-personal-access-token
# give token repo/public_repo permission
ACCESS_TOKEN = open('GitKey.txt').read().strip()
g = Github(ACCESS_TOKEN)

fout_result = open("F:/University/SoSu 2022/MSR/GitHub_Repo/search_github_allcontribs_results.txt", 'a')
test_counter = 0
done = False
sizeL, sizeU = size, size + step_size


# main loop
while True:
    # probe to check if we're done
    if test_counter == probe_trigger:
        query = f'filename:.all-contributorsrc path:/ size:>{size}'
        result, num_results = post_query(query)
        if num_results == 0:
            print("[Status]", "Finished scan at", size, "bytes, no file left.")
            done = True
        test_counter = 0  # reset counter
    if done:
        break

    # get next slice of results
    query_elem = ['filename:.all-contributorsrc',  # all-contributors files
                  "path:/",                        # only in base directory
                  f'size:{sizeL}..{sizeU}',        # slice by size
                  'sort:indexed']                  # sort (to avoid dupes)
    query = " ".join(query_elem)
    print("[Query]", query)
    result, num_results = post_query(query)
    print("[Status]", "Found", num_results, "files.")

    # act on slice of results
    if num_results >= 1000 and step_size > 1:
        # resize query if too many results
        step_size = max(int(step_size / multiplier), 1)
        sizeL, sizeU = size, size + step_size
        print("[Status]", "Too many results, resizing step_size to", step_size)
        continue
    else:
        # process query
        
        
        size += step_size
        if num_results >= 1000 and step_size == 1:
            print("[Warning]", "Dropping results due to search space overflow.")
        if num_results > 0:
            print("[Status]", "Processing", num_results, "files in range", sizeL, "..", sizeU)
            exhaust_list(result, sizeL, sizeU, fout_result)
        # possibly adapt the search strategy
        if num_results <= min_result:
            step_size = int(multiplier * step_size)
            print("[Status]", "Too few results, resizing step_size to", step_size)
        # udapte counters
        test_counter += 1
        sizeL, sizeU = size, size + step_size

# Exit
fout_result.close()






