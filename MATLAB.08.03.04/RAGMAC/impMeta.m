function impMeta(fileName,pos)
set(gcf, 'PaperPositionMode', 'auto');
set(gcf,'Units','centimeters')
set(gcf,'Position',[0 0 pos])


cd pics
fileName = [fileName '.emf'];
print('-dmeta',fileName)
close all
%winopen(fileName)
cd ..