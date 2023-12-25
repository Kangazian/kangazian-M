function [x_gbest,best_so_far] = evalute_gbest(position,fittness,max_fit,opt_sol,best_so_far,it)
if it == 1
    prefittness = fittness;
    preposition = position;
    best_so_far = max_fit;
elseif max_fit>=best_so_far(it-1);
        best_so_far(it) =max_fit;
    else
        best_so_far(it) = best_so_far(it-1);
end
x_gbest = opt_sol;
return
