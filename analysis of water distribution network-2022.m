%%%%%%%%%%%%%%%%%%%%%%%in the name of God%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             analysis of water distribution of network using      %                
%                   Linear theory method(MATLAB)                   % 
%                                                                  % 
%                                                                  %                                           
%                  prepare by :mohammad. kangazian                 %    
%                        2014                                      % 
%                                                                  %                        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear 
%assume
%D=Dimeter(m),l=lenght(m),z1=elevation(m),Q=discharge(m^3/s)%
D=[.4582,.2036,.4064,.1018,.4064,.3055,.2036,.2036];
l=[1000,1000,1000,1000,1000,1000,1000,1000];
z1=[150,160,155,150,165,160,150,150];
%
%initial discharge (m^3/s)%
Q=[1 1 1 1 1 1 1 1];

%y=slope ,E=roughness%
E=.00025; ;y=.000001;

%friction= result of friction coefficient matrix%
%result=result of discharge (Q m^3/h) matrix%
friction=size(100,8)
result =size(100,8)

for i=1:8 
     result(1,i)=Q(i)
end

% EQ-Colebroook(1938) for turbulent flow R>=4000%

for t=1:8
f(t)=1.325*((log((E/(3.7*D(t)))+(4.618*((y*D(t))/Q(t))^.9)))^-2)
end

for i=1:8
    friction(1,i)=f(i)
end

% coefficient resistance in pipe%

for i=1:8
   r(i)=(8*f(i)*l(i))/((pi^2)*9.806*D(i)^5)
end

% n=constant; for equation Darci-v n=2%
 n=2;
 
 %cal -R
 
 for i=1:8
     R(i)=r(i)*(abs(Q(i)))^(n-1);
 end
 
 % a=matrix of network%
 
 a=[1 -1 -1 0 0 0 0 0
   0 1 0 0 0 0 -1 0
   0 0 1 -1 -1 0 0 0
   0 0 0 1 0 0 1 1
   0 0 0 0 1 -1 0 0
   0 0 0 0 0 1 0 -1
   0 -R(2) R(3) R(4) 0 0 -R(7) 0
   0 0 0 -R(4) R(5) R(6) 0 R(8)];

%b=matrix of demand(m^3/h)

b=[100;100;120;270;330;200;0;0]; 

% Q1=result of discharge in iteration 1 (m^3/s)%

Q1=(inv(a)*b/3600)

for i=1:8
    result(2,i)=Q1(i)
end
% mian=average of discharge in iteration (i & i+1) matrix%

mian=size(100,8);

for counter =3:40
for o=1:8
mian(o) = (result(counter-1,o) + result(counter-2 , o) )*.5
end

% EQ-Colebroook(1938) for turbulent flow R>=4000 in average discharge%

for t=1:8
f1(t)=(1.325*((log((E/(3.7*D(t)))+(4.618*((y*D(t))/mian(t))^.9)))^-2))
end

for i=1:8
    friction(2,i)=f1(i)
end

% coefficient resistance in pipe in average discarge%

for i=1:8
    r1(i)=((8*f1(i)*l(i))/((pi^2)*9.806*D(i)^5))
end

 for i=1:8
      R1(i)=r1(i)*(abs(mian(i)))^(n-1);
 end
 
 %a=matrix of network in iteration  in average discharge%
 a=[1 -1 -1 0 0 0 0 0
   0 1 0 0 0 0 -1 0
   0 0 1 -1 -1 0 0 0
   0 0 0 1 0 0 1 1
   0 0 0 0 1 -1 0 0
   0 0 0 0 0 1 0 -1
   0 -R1(2) R1(3) R1(4) 0 0 -R1(7) 0
   0 0 0 -R1(4) R1(5) R1(6) 0 R1(8)];

%cal- discharge for average %

Qf=(inv(a)*b/3600)

for i=1:8
    result(counter,i)=Qf(i)
end

%fQ=final friction coefficient in pipes%

for t=1:8
fQ(t)=1.325*((log((E/(3.7*D(t)))+(4.618*((y*D(t))/Qf(t))^.9)))^-2)
end

 for i=1:8
    friction(counter,i)=fQ(i)
end

Q=Qf
end

finalfriction=fQ

%Qfinal=final discharge in pipes%

Qfinal=Qf*3600

