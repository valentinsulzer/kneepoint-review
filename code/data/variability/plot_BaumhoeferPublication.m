load('lifetime.mat')

cell_fieldnames = fieldnames(lifetime);

figure('Name','Capacity');
hold on;


for cell_counter=1:size(cell_fieldnames,1)
    plot(lifetime.(cell_fieldnames{cell_counter}).cyc,lifetime.(cell_fieldnames{cell_counter}).cap,'Color',[0,0.33,0.63])
end

clear cell_counter

xlim([0 1800]);
ylim([0.8 1.9]);
ylabel('Remaining capacity in Ah');
xlabel('Cycles');
box;
grid on;

line([0 1800],[1.48 1.48],'Color',[0.34,0.67,0.15],'LineStyle','--')
line([0 1800],[1.2 1.2],'Color',[0.34,0.67,0.15],'LineStyle','--')

clear cell_fieldnames lifetime