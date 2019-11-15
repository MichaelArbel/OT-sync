
command = 'C:\Users\tolga\source\repos\Super4PCS\x64\Release\Super4PCS.exe -i d:/Data/model_normalized.obj D:\Data\meas_sync\kbest\ShapeNet2Samples\model0\7c8a7f6ad605c031e8398dbacbe1f3b1\point_cloud';

Qabs = [];
for i=0:39
    curCommand = sprintf('%s/%d.obj',command,i);
    system(curCommand);
    M = load('output.pose');
    R = M(1:3,1:3);
    q = R2q(R);
    Qabs=[Qabs; t_qconj(q)];
end
writematrix(Qabs, 'D:\Data\meas_sync\kbest\out\shapenet_Qabs.txt');
writematrix(Edges, 'D:\Data\meas_sync\kbest\out\shapenet_Edges.txt');