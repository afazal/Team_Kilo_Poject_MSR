# file locations
EXTACTION_SCRIPT=$PWD"/scrape-allcontribs-history.py"
GIT_FOLDERS=/storage/MSR2021/git_folders/

SCRATCH1=../data/scratch_scraper.txt
SCRATCH2=../data/scratch_output.txt

#  clean state
for REPO in $(ls $GIT_FOLDERS); do
    git checkout HEAD -- .all-contributorsrc
done
# loop over repos
for REPO in $(ls $GIT_FOLDERS); do
    # get git log
    cd $GIT_FOLDERS/$REPO
    git log --date=local --pretty=format:"%H04%x09%ad%x09" > $SCRATCH1
    echo "[Entering]" $PWD

    # replay history with git
    touch $SCRATCH2
    while read SHA DATE; do

        echo "[Processing]" $SHA $DATE
        git checkout ${SHA:0:20} -- .all-contributorsrc 
        if [ "$?" -ne 0 ]; then
            echo "[Stopping] Reached origin."
            break
        fi
        echo -e "#begin\t"$SHA"\t"$DATE>> $SCRATCH2
        cat .all-contributorsrc >> $SCRATCH2
        echo " " >> $SCRATCH2
        echo "#end "$SHA >> $SCRATCH2
    done < $SCRATCH1
    git checkout HEAD -- .all-contributorsrc  

    # extact and pickle history into root
    python $EXTACTION_SCRIPT
    mv "../data/scratch_pickled.pck" $GIT_FOLDERS/$REPO"/__contributor_history.pck"
    rm $SCRATCH2
done