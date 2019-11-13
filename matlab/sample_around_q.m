
function [Q] = sample_around_q(q, N, eucSampleMode)

if (eucSampleMode==0)
    Q = repmat(q,N,1) + 0.01*randn(N,4 );
else
    Q = repmat(q,N,1) + 0.01*rand(N,4);
end

Q = Q./repmat(sqrt(dot(Q,Q,2)), 1,4);

end