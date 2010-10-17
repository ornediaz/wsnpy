% this files computes GEM as a function of the number of retransmissions
function gem
clc,close all
p=0.3;
maxNumber=20;%maximum number of retransmissions


[cost,reliability] = retransmissionCombinations(p,maxNumber)

subplot(2,1,1)
plot(reliability,cost,'-*')
xlabel('Reliability'),ylabel('Cost')
title('Cost and reliability for different values of the maximum number of retransmissions')

%axis([0 1 0 max(cost)])

subplot(2,1,2)
r2c=reliability./cost;
stem(r2c)
xlabel('Maximum number of transmissions')
title('Quotient between reliability and cost')

function [cost,reliability]=retransmissionCombinations(p,maxNumber)
clc
if 0
    tic
    v=0:maxNumber-1;
    q=(1-p).^(0:maxNumber-1);
    reliability=p*cumsum(q);
    cost =cumsum(q);
    toc
else
    tic
    a=zeros(1,maxNumber);
    b=a;
    for g=1:maxNumber
        for k=1:g
            y=p*(1-p)^(k-1);
            a(g)=a(g)+y;
            b(g)=b(g)+k*y;
        end
        b(g)=b(g)+g*(1-p)^k;
    end
    reliability=a;
    cost=b;
    toc
end
