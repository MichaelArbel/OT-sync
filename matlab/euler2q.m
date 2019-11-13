
function [q] =euler2q(eu)

q = R2q(euler2rot(eu));

end