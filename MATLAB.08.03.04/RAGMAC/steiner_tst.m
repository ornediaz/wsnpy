% Short driver function to test steiner.m
clear,clc,close all,rand('twister', 5489);
N = 20;

x = 15;
y = 30;
txRg = 10;
ixRg = 21;

P=[ 0 0 ;rand(N-1,2) .* repmat([x y],N-1,1)];

te1 = tree([x y txRg ixRg],P);
te1.src = 11:20;
te1.nPkts = 4;
subplot(1,3,1)
plot(te1);
subplot(1,3,2)
te2 = prune(te1);
plot(te2)
te3 = steiner(te1);
subplot(1,3,3)
plot(te3)