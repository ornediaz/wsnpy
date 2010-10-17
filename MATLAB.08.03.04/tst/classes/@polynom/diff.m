function q = diff(p)
% POLYNOM/DIFF  DIFF(p) is the derivative of the polynom p.
c = p.c; 
d = length(c) - 1;  % degree
q = polynom(p.c(1:d).*(d:-1:1));