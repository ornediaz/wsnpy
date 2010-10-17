function y = end(p,k,n)
if (n==1 & k==1)
	y=length(p);
else
	error('The queue has only dimension 1')
end
	