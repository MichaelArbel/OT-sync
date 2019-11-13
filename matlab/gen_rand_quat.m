
function [Q] = gen_rand_quat(N, eucSampleMode)

if (eucSampleMode==0)
    d   = randn( [4, prod( N )] );
else
    d   = rand( [4, prod( N )] );
end
n   = sqrt( sum( d.^2, 1 ));
dn  = bsxfun( @rdivide, d, n );
neg = dn(1,:) < 0;
dn(:,neg) = -dn(:,neg);
Q   = [dn(1,:); dn(2,:); dn(3,:); dn(4,:) ]';

end