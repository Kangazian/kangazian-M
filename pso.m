'********prepared by mohamad kangazian******'
clc
clear all
close all
np = 8;
iteration = 40;
var = 2;
max_x = [2 1];
min_x = [-1 -2];
prefittness = [];
preposition = [];
c1 = 1.5;
c2 = 1.5;
w_initial = 0.4;
w_final = 0.2;
best_so_far = [];
average_fit = [];
v_max = 0.3*(max_x-min_x);
v = [];
w = [];

%%%%%%%initial population%%%%%%%%%%%%%%%
for k = 1:var
    position(:,k) = min_x(k) + (max_x(k)-min_x(k))*rand(np,1);
end
position;
%%%%%%%%%%%%%%%%%%%%%%%
for it = 1:iteration
    [max_fit,opt_sol,fittness,avr_fit] = fit_eval(position,np);
    average_fit(it) = avr_fit;
    [x_gbest,best_so_far] = evalute_gbest(position,fittness,max_fit,opt_sol,best_so_far,it);
    [x_pbest,prefittness,preposition] = evalute_pbest(fittness,position,prefittness,preposition,np,it);
    if it == 1
        w(it) = w_initial;
    else
        w(it) = w_initial - ((w_initial-w_final)/iteration)*it;
    end
    w1 = w(it); 
    [position,v] = new_position(x_gbest,x_pbest,position,c1,c2,v_max,var,np,it,max_x,min_x,v,w1);
end

%%%%%%%%%%%%display%%%%%%%%%%%%%%%%
display('final solution    optimum fitness');
result=[opt_sol,max_fit];
x = 1:iteration;
plot(x,best_so_far,'.-b',x,average_fit,'*-r');
xlabel('generation');
ylabel('fitness function');
legend('best-so-far','average fitness');
% figure(2)
% plot(position,'.r')
% hold
% plot(preposition,'.b')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
