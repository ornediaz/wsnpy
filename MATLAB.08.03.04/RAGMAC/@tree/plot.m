function plot(te)
% Plot tree marking nodes & links
plot(te.P(1,1),te.P(1,2),'o')
hold on
for k = 1:te.N
	if te.pt(k) > 0
		x = te.P(k,1);
		y = te.P(k,2);
		if sum(te.src == k)
			plot(x,y,'ok')
		else
			plot(x,y,'+')
		end
		p1 = te.P(k,:);
		p2 = te.P(te.pt(k),:);
		plot([p1(1) p2(1)],[p1(2) p2(2)])
		text( p1(1) + .01 * te.x , p1(2) , num2str(k) )
	end
end
axis( [ -.1 * te.x, te.x * 1.1 , -.1 * te.y , te.y * 1.1 ] )
axis equal
xlabel('x axis (m)')
ylabel('y axis (m)')