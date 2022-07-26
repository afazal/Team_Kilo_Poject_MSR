#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is used to Scrape repositories for meta data like language.

from github import Github
from github.GithubException import RateLimitExceededException, UnknownObjectException
import json
import time
import pathlib


def community_profile(repository):
    profile = dict()
    # files the n basepath
    base_content = {file.name.lower() for file in repository.get_contents("/")}
    for file in ["code_of_conduct", "readme", "contributing", "license"]:
        profile[file] = False
        for ext in ["", ".md", ".rst", ".txt"]:
            if file + ext in base_content:
                profile[file] = True
    # files in /.github
    try:
        github_content = {file.name.lower() for file in repository.get_contents("/.github")}
    except UnknownObjectException:
        github_content = set()
    except TypeError:  # .github is a file
        github_content = set()

    for file in ["issue_template", "pull_request_template"]:
        profile[file] = False
        for ext in ["", ".md", ".rst", ".txt"]:
            if file + ext in github_content:
                profile[file] = True
    # description
    profile['description'] = repository.description is not None
    return profile


# boot up github instance
with open("secret.txt", "r") as f:
    token = f.readline().strip()  
g = Github(token)


with open("repo_list.txt", "r") as repo_list:
    for repo in repo_list:
        # # check if scraping already done
        outfile_name = "../data/gh-metadata/" + repo.strip().replace("/", "_") + ".json"
        if pathlib.Path(outfile_name).is_file():
            print("[Skipping]", repo.strip())
            continue
        # query meta-data
        print(repo.strip())
        while True:
            try:
                r = g.get_repo(repo.strip())
                repo_data = {'languages': r.get_languages(),
                             'profile': community_profile(r),
                             'stars': r.stargazers_count,
                             'forks': r.forks_count,
                             'commits': r.get_commits().totalCount}
                # save to file
                with open(outfile_name, "w") as outfile:
                    json.dump(repo_data, outfile, indent=2)
                break
            except RateLimitExceededException:
                print("[RateLimitExceededException] Sleeping for 60 seconds.")
                time.sleep(60)
                continue
            except UnknownObjectException:
                print("[UnknownObjectException]", repo.strip(), "doesn't exist.")
                break
