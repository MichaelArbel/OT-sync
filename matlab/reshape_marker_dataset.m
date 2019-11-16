
folder = 'D:\Data\meas_sync\marker1';
N = 9;
% 
I = [];
fid = fopen([folder '/QRel.txt'], 'w');
fidEdges = fopen([folder '/Edges.txt'], 'w');
for i=0:N
    for j=0:N
        if (i~=j)
            fn = sprintf('%s/out/quat_%02d_%02d.txt', folder, i, j);
            
            if (exist(fn,'file'))
                Q = load(fn)';
                I = [I; int32([i,j])];
                
                fprintf(fid, '%g ', Q(:));
                fprintf(fid, '\n');
                fprintf(fidEdges, '%d %d\n', i+1, j+1);
            end
            
        end
    end
end
fclose(fid);
fclose(fidEdges);

fnAbs = 'D:\Data\meas_sync\marker1\marker_Qabs.txt';
fnRel = 'D:\Data\meas_sync\marker1\marker_Qrel.txt';
fnEdges = 'D:\Data\meas_sync\marker1\Edges.txt';


fnEdges = 'C:\Users\tolga\Box Sync\gitforks\OptimalTransportSync\py\marker_Edges.txt';
Edges = load (fnEdges);
Edges = I+1;

fnAbs = 'C:\Users\tolga\Box Sync\gitforks\OptimalTransportSync\py\marker_Qabs.txt';
fnRel = 'C:\Users\tolga\Box Sync\gitforks\OptimalTransportSync\py\marker_Qrel.txt';
Q = load (fnAbs);
Qabs = Q;
Qrel = load (fnRel);
qrelInsert = t_qconj(Qabs(2,:));
Qrel = [qrelInsert qrelInsert qrelInsert qrelInsert; Qrel];
qrelInsert = (Qabs(2,:));
Qrel = [qrelInsert qrelInsert qrelInsert qrelInsert; Qrel];

% insert a camera
Qabs = [1 0 0 0; Qabs];
Edges = Edges + 1;
Edges = [[1 2]; Edges];

writematrix(Qabs, fnAbs, 'Delimiter', ' ');
writematrix(Qrel, fnRel, 'Delimiter', ' ');
writematrix(Edges, fnEdges, 'Delimiter', ' ');

for i=1:size(Q,1)
    q = Q(i, :);
    for j=1:4:length(q)
        ind = j:j+3;
        qc = q(ind);
        qc = t_qconj(qc);
        Q(i, ind) = qc;
    end
end
writematrix(Q, fnAbs, 'Delimiter', ' ');


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