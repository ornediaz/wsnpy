function plt(P,previous,xSide,ySide)
clf
plot(P(1,1),P(1,2),'o')
hold on
for x = 1:length(P)
	if previous(x) > 0
		plot(P(x,1),P(x,2),'o')
		p1=P(x,:);
		p2=P(previous(x),:);
		plot([p1(1) p2(1)],[p1(2) p2(2)])
		text(p1(1)+.01*xSide,p1(2),num2str(x))
	end
end
axis([-.1*xSide xSide*1.1 -.1*ySide ySide *1.1])
