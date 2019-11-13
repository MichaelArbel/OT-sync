function [d] = sinkhornDistQ(Q1, Q2)

lambda = 0.05;
tolerance = 0.00001;
maxIter = 1000;

N1 = size(Q1,1);
N2 = size(Q2,1);
M = zeros(N1, N2);
for i=1:N1
    for j=1:N2
        M(i,j) = t_qdist(Q1(i,:), Q2(j,:));
    end
end

a = ones((N1), 1)./N1;
b = ones((N2), 1)./N2;
%a=1./N1;
%b=1./N2;

tic();
K = exp(-lambda.*M);
U = K.*M ;
[d,~,~,~]=sinkhornTransport(gpuArray(a), gpuArray(b), gpuArray(K), gpuArray(U),lambda,[],[],tolerance,maxIter);
toc();



end