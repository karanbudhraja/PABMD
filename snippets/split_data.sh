#!/usr/bin/env bash

# get header
head processed_forward_kinematics_a.txt -n 1 > header_a.txt
head processed_forward_kinematics_b.txt -n 1 > header_b.txt

# get remaining file
tail processed_forward_kinematics_a.txt -n +2 > headerless_a.txt
tail processed_forward_kinematics_b.txt -n +2 > headerless_b.txt

# split into train and test
# 50% based on paper
split -dl $[ $(wc -l headerless_a.txt | cut -d" " -f1) * 50 / 100 ] headerless_a.txt headerless_a_split_
mv headerless_a_split_00 headerless_a_train
mv headerless_a_split_01 headerless_a_test
split -dl $[ $(wc -l headerless_b.txt | cut -d" " -f1) * 50 / 100 ] headerless_b.txt headerless_b_split_
mv headerless_b_split_00 headerless_b_train
mv headerless_b_split_01 headerless_b_test

# a has sizes 1500
# b has sizes 2000
rm train/* test/*
split -dl 1500 headerless_a_train train/forward_kinematics_a_train_set_ -a 4 --additional-suffix=.txt
split -dl 2000 headerless_b_train train/forward_kinematics_b_train_set_ -a 4 --additional-suffix=.txt
split -dl 1500 headerless_a_test test/forward_kinematics_a_test_set_ -a 4 --additional-suffix=.txt
split -dl 2000 headerless_b_test test/forward_kinematics_b_test_set_ -a 4 --additional-suffix=.txt

# construct train data with headers
for fileName in train/*_a_*.txt; do
    cat header_a.txt $fileName > temp
    mv temp $fileName
done
for fileName in train/*_b_*.txt; do
    cat header_b.txt $fileName > temp
    mv temp $fileName
done

# clean up
rm *header*
