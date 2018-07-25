#!/bin/sh

# export TrueSkillTemp  = ~/Documents/trueskill_temp/
# mkdir /mnt/cluster
# cd /mnt/cluster/PGMS/scoring/
# sudo mount -t cifs //192.168.0.100/cluster /mnt/cluster
# python trueskill_score.py $1 $2

!rsync -av --progress /mnt/cluster/scoring /home/ipycluster/Documents/ --exclude .git --exclude elo_temp_goals/ --exclude trueskill_temp/ --exclude trueskill_data_for_classifier/ --exclude elo_goals_data_for_classifier/
