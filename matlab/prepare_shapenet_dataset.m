
folder = 'C:\Users\tolga\Box Sync\gitforks\OptimalTransportSync\py\MMDSync\out';

Qrel = [];

Edges = [];
N = 39;
for i=0:N
    for j=0:N
        fn = sprintf('%s/%d.obj_%d.obj.pose', folder, i, j);
        if (exist(fn, 'file'))
            M = load(fn);
            q = R2q(M(1:3,1:3));
            Qrel = [Qrel; t_qconj(q)];
            Edges = [Edges; [i j]+1];
        end
    end
end
 
% absolute
absfolder = 'D:\Data\meas_sync\kbest\ShapeNet2Samples\model0\7c8a7f6ad605c031e8398dbacbe1f3b1\point_cloud';
Qabs = [];
for i=0:N
    fn = sprintf('%s/%d_rotate.txt', absfolder, i);
    if (exist(fn, 'file'))
        M = load(fn);
        q = R2q(M(1:3,1:3));
        Qabs = [Qabs; t_qconj(q)];
    end
end

writematrix(Qrel, 'D:\Data\meas_sync\kbest\out\shapenet_Qrel.txt');
writematrix(Qabs, 'D:\Data\meas_sync\kbest\out\shapenet_Qabs.txt');
writematrix(Edges, 'D:\Data\meas_sync\kbest\out\shapenet_Edges.txt');