#!/bin/sh

rm -rf ./arxiv-submission/;

mkdir ./arxiv-submission;
cp main.tex arxiv-submission/;
cp main.bbl arxiv-submission/;
cp abstract.tex arxiv-submission/;
cp introduction.tex arxiv-submission/;
cp formulation.tex arxiv-submission/;
cp mdp.tex arxiv-submission/;
cp benchmarking.tex arxiv-submission/;
cp experiments.tex arxiv-submission/;
cp discussion.tex arxiv-submission/;
cp extensions.tex arxiv-submission/;
cp more-experiments.tex arxiv-submission/;
cp proof-limit-experiment.tex arxiv-submission/;
cp proof-mdp.tex arxiv-submission/;
cp proof-asymptotic.tex arxiv-submission/;
cp implementation.tex arxiv-submission/;

cd ./arxiv-submission/;
mkdir macros;
cd macros;
cp ../../../macros/statistics-macros.sty ./;
cp ../../../macros/packages.sty ./;
cp ../../../macros/formatting.sty ./;
cd ../../

cd ./arxiv-submission/;
mkdir fig;
cd fig;
cp ../../fig/* ./
