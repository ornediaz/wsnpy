function imp(fileName,PaperPosition)
if nargin < 2
	PaperPosition = [0 0 9.5 5];
end
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition',PaperPosition );
cd pics
print('-depsc',[fileName '.eps'])
print('-dmeta',[fileName '.emf'])
print('-dtiff','-r300',fileName)
saveas(gcf,[fileName '.fig'])
cd ..