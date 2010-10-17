function display(eL)
disp('=======================')
disp('Current event list')
for kk = 1:length(eL.time)
	fprintf('t = %7.4f, x = %3.0f, type = %s \n',...
		eL.time(kk) , eL.node(kk),eL.type{kk})
end
disp('=======================')