function [x_pbest,prefittness,preposition] = evalute_pbest(fittness,position,prefittness,preposition,np,it)
x_pbest = [];
for i = 1:np
    if it == 1
        x_pbest = position;
    elseif fittness(i)>prefittness(i);
        x_pbest(i,:) = position(i,:);
    else
        x_pbest(i,:) = preposition(i,:);
    end
    prefittness = fittness;
    preposition(i,:) = position(i,:);
end
return