
function [R] = euler2rot(eu)

phi = eu(1);
theta = eu(2);
psi = eu(3);

cpsi = cos(psi);
spsi = sin(psi);

ctheta = cos(theta);
stheta = sin(theta);

cphi = cos(phi);
sphi = sin(phi);

R = zeros(3,3);

R(1,1) = (  cpsi*cphi  -  spsi*ctheta*sphi  );
R(1,2) = ( -cpsi*sphi  -  spsi*ctheta*cphi );
R(1,3) = (  spsi*stheta );

R(2,1) = (  spsi*cphi  +  cpsi*ctheta*sphi );
R(2,2) = ( -spsi*sphi  +  cpsi*ctheta*cphi );
R(2,3) = ( -cpsi*stheta );

R(3,1) = (  stheta*sphi );
R(3,2) = (  stheta*cphi );
R(3,3) = (  ctheta);

end