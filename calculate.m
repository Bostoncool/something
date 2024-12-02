%%%%%%%%%% 以下是“多河段 BOD-DO 耦合矩阵模型”的程序代码%%%%%%%%%%
function F=calculate(N,Q1,Q,L,O,kd,ka,t,Q3,T,L20,O20)
% Q-在断面i处注入河流的流量,m3/s
% L，O-由断面i注入河流的污水的污染物（例如BOD）浓度与溶解氧（DO）浓度,mg/L
% kd-BOD的降解速度常数,/d
% ka-大气复氧速度,/d
% t-由断面i到断面i+1经过时间，d
% Q3-在断面i处引出的流水流量,m3/s
% L20,O20-河流背景的BOD和DO浓度，mg/L
% 输入题目中变量值
 
Os=468/(31.6+T);%饱和氧算法
for i=1:1:N
 Q2(i,1)=Q1(i,1)-Q3(i,1)+Q(i,1); %河流Q和BOD的平衡关系，连续性原理
 Q1(i+1,1)=Q2(i,1);
end
Q1; % Q1i-由上一个河段流到断面i的河水流量，mg/L
Q2; % Q2i-由断面i向下游河段流出的河水流量，mg/L
Q3;
for i=1:1:N
 a(i,1)=exp(-kd(i,1)*t(i,1));%S-P模型中BOD的变化规律，阿尔法α改为了a
end
a;
for j=1:N
    for i=1:N
     if j==i
        A(i,j)=1;%中间一个穿孔全是1
     else if j==i-1%穿孔朝下是正常运算的值
             A(i,j)=-a(i,1)*(Q1(i,1)-Q3(i,1))/Q2(i,1);%A矩阵有效值的计算
         else A(i,j)=0;
         end
      end
   end
end
for i=1:N
     for j=1:N
         if i==j
            B(i,j)=Q(i,1)/Q2(i,1);%B矩阵有效值的计算
         else B(i,j)=0;
         end
      end
end
A;
B;
g=zeros(N,1);
g(1,1)=a(1,1)*(Q1(1,1)-Q3(1,1))/Q2(1,1)*L20;% 零维矩阵的建立，书上公式
g;
for i=1:N
y(i,1)=exp(-ka(i,1)*t(i,1));
end
y;
for j=1:N
    for i=1:N
        if j==i
           C(i,j)=1;
        else if j==i-1
                C(i,j)=(Q1(i,1)-Q3(i,1))/Q2(i,1)*-y(i,1);
           else C(i,j)=0;
           end
        end
     end
end
for j=1:N
    for i=1:N
        if j==i-1
           D(i,j)=(Q1(i,1)-Q3(i,1))/Q2(i,1)*kd(i,1)/(ka(i,1)-kd(i,1))*(a(i,1)-y(i,1));
        else D(i,j)=0;
        end
     end
end
C;
D;
for i=1:N
    f(i,1)=(Q1(i,1)-Q3(i,1))/Q2(i,1)*Os*(1-y(i,1));
end
f;
h=zeros(N,1);
h(1,1)=(Q1(1,1)-Q3(1,1))/Q2(1,1)*y(1,1)*O20-(Q1(1,1)-Q3(1,1))/Q2(1,1)*kd(1,1)/(ka(1,1)-kd(1,1))*(a(1,1)-y(1,1))*L20;
h;
U=A^-1*B; %U是BOD对BOD响应矩阵
U;
V=-C^-1*D*A^-1*B; %V是溶解氧对BOD的响应矩阵
V;
m=A^-1*g;
m;
n=C^-1*B*O+C^-1*(f+h)-C^-1*D*A^-1*g;
n;
L2=U*L+m;
L2;
O2=V*L+n;
O2;
res=cell(1,6);
res{1}=U;
res{2}=V;
res{3}=m;
res{4}=n;
res{5}=L2;
res{6}=O2;
F=res;
 
 