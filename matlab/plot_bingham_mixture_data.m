function plot_bingham_mixture_data(data,quality)

clf;

%subplot(2,1,1);
[SX,SY,SZ] = sphere(quality);
n = size(SX,1);

z=30;
B.d = 4;

% compute the marginal distribution of the axis 'u'
C = zeros(n);
Z = -[z z z];
B.Z = Z;
F = bingham_F(B.Z);

Q = data.q;
scale = 15;

[I,J] = meshgrid(1:n,1:n);

for k=1:length(Q)
    q = Q(k,:);
    V = compute_V_frames(q);
    B.V = V;
    Z = data.z(k, :);
    B.Z = Z;
    w = data.w(k);
    for id=1:length(I(:))
        i = I(id);
        j = J(id);
        u = [SX(i,j) SY(i,j) SZ(i,j)];
        for a=0:.1:2*pi
            qs = [cos(a/2), sin(a/2)*u];
            %d = t_qdist(qs,q);
            %V = compute_V_frames(qs);
            %z = 1./d;
            %
            C(i,j) = C(i,j) + scale*w*bingham_pdf_3d(qs, Z(1),Z(2),Z(3), V(:,1), V(:,2), V(:,3), F);
        end
    end
end

C = C./max(max(C));
%C = .5*C + .5

surf(SX,SY,SZ,C, 'EdgeColor', 'none', 'FaceAlpha', .7);
axis vis3d;
axis off;
%colormap(.5*gray+.5);
cmap = pink;
cmap(1:2:end,:) = cmap(end/2+1:end,:);
cmap(2:2:end,:) = cmap(1:2:end,:);
cmap = .75*cmap + .15*autumn + .1*gray;
colormap(jet);


if nargin >= 2
   hold on, plot_quaternions(Q);
end