function dijkstra
%% Centralized Dijkstra's algorithm
% It is complete and works
%% Create the points' coordinates
clear,clc,close all
N=101;
P=rand(N,2);
plot(P(:,1),P(:,2),'o')
hold on

%% Initialize
INF=100;

d=ones(1,N)*INF;
d((N+1)/2)=0;
previous=-1*ones(1,N);
% If element 'i' of temp is zero, it means it has not been processed yet
temp=zeros(1,N);
for kk=1:N %For all vertices
    [m u]=min(d+temp*2*INF);
    temp(u)=1;
    for jj=[1:u-1 u+1:N] % For each neighbor 'jj' of 'u'
		dist=norm(P(u,:)-P(jj,:)); % distance to 'jj' from 'u'
		if dist > .3
			dist=INF;
		end
        alt=d(u)+dist;
        if (alt < d(jj))
            d(jj) = alt;
            previous(jj) = u;
        end
    end 
end
%% Plot the tree
for kk=1:N
	if previous(kk) > 0
		p1=P(kk,:);
		p2=P(previous(kk),:);
	plot([p1(1) p2(1)],[p1(2) p2(2)])
	end
end

%%
function routeWithFailure
clc
% 4 vertices
    %cost=diag([3 1 1],1)+diag([1 2],2)+diag(5,3)
% Construct the matrix with the cost between any two nodes
cost=construct([3 1 5 1 2 1]) ;
[d,previous]=Dijkstra(cost);

function [d,previous]=Dijkstra(cost)
INF=100;
N=length(cost);
d=ones(1,N)*INF;
d(1)=0;
previous=-1*ones(1,N);
temp=ones(1,N);
for kk=1:N %For all vertices
    [m u]=min(d+(~temp)*2*INF);
    temp(u)=0;
    for jj=[1:u-1 u+1:N] % For each neighbor 'jj' of 'u'
        alt=d(u)+cost(u,jj); % distance to 'jj' from 'u'
        if (alt < d(jj))
            d(jj) = alt;
            previous(jj) = u;
        end
    end 
end

function cost = construct(v)
% Construct adjacecy matrix
x=length(v);
N=(1+sqrt(1+8*x))/2;
if (N~=round(N))
    error('The lenght of the input vector to "construct" is incorrect')
end
cost = zeros(N);
punt=0;
for kk=1:N-1;
    num=N-kk;
    cost(kk,kk+(1:num))=v(punt+(1:num));
    punt=punt+num;
end
cost=cost+cost';
% vim:foldmethod=marker:ts=4:sw=4:tw=76
