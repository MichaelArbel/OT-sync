
folder = 'D:\Data\meas_sync\marker1';
N = 10;

I = [];
fid = fopen([folder '/QRel.txt'], 'w');
fidEdges = fopen([folder '/Edges.txt'], 'w');
for i=1:N
    for j=1:N
        if (i~=j)
            fn = sprintf('%s/out/quat_%02d_%02d.txt', folder, i, j);
            
            if (exist(fn,'file'))
                Q = load(fn);
                I = [I; int32([i,j])];
                
                fprintf(fid, '%g %g %g %g\n', Q);
                fprintf(fid, '\n');
                fprintf(fidEdges, '%d %d\n', i, j);
            end
            
        end
    end
end

fclose(fid);
fclose(fidEdges);
%save([folder '/Edges.txt'], 'I', '-ascii');

Q=[ 3.55603171e-30, -1.18534390e-30,  7.07106781e-01,  7.07106781e-01
  7.07106781e-01, -7.07106781e-01,  4.67221225e-18, -4.67221225e-18
  0.5, -0.5, -0.5, -0.5
 -2.55341042e-15,  2.47374606e-15,  7.07106781e-01,  7.07106781e-01
 -1.65167860e-15,  1.65649523e-15,  7.07106781e-01,  7.07106781e-01
  7.07106781e-01, -7.07106781e-01, -2.47374606e-15, -2.55341042e-15
  7.07106781e-01, -7.07106781e-01, -1.65649523e-15, -1.65167860e-15
 -2.29562359e-16,  2.20586461e-15,  7.07106781e-01,  7.07106781e-01
 -1.36464812e-16,  1.76447534e-15,  7.07106781e-01,  7.07106781e-01
 -7.07106781e-01,  7.07106781e-01,  2.20586461e-15,  2.29562359e-16
  2.29562359e-16, -2.20586461e-15,  7.07106781e-01,  7.07106781e-01
 -7.07106781e-01,  7.07106781e-01, -1.76447534e-15, -1.36464812e-16];

data = load('C:\Users\tolga\Downloads\dress_4.mat');
Q=data.q;
for i=1:length(Q)
    q = Q(i, :);
    if (q(1)<0)
        q=-q;
        Q(i,:) = q;
    end
end
50
figure, plot_bingham_mixture(Q,50);