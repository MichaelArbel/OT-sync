
data = load('C:\Users\tolga\Downloads\test_relative\test_relative.mat');

n = length(data.pred_all_q);
d = size(data.entropy_r,2);

% make graph
Edges = [];
for i=0:n-1
    i1 = mod(i-1, n);
    i2 = mod(i+1, n);
    Edges = [Edges [i i1]'];
    Edges = [Edges [i i2]'];
end
Edges = Edges + 1;

QrelHE = [];
QrelLE = [];
for k=1:length(Edges)
    i1 = Edges(1,k);
    i2 = Edges(2,k);
    
    Qs1 = squeeze(data.pred_all_q(i1,:,:));
    Qs2 = squeeze(data.pred_all_q(i2,:,:));
    
    % HE
    HE = [];
    for i=1:d
        R1 = q2R(Qs1(i,:));
        for j=1:d
            if (i~=j)                
                R2 = q2R(Qs2(j,:));
                R = R1*R2';
                qr = R2q(R);
                HE = [HE qr];
            end
        end
    end
    QrelHE = [QrelHE; HE];
    
    % LE
    LE = [];
    for i=1:d
        R1 = q2R(Qs1(i,:));
        R2 = q2R(Qs2(i,:));
        R = R1*R2';
        qr = R2q(R);
        LE = [LE qr];
    end
    QrelLE = [QrelLE; LE];
end

writematrix(data.labels(:,1:4), 'D:\Data\blue_chairs_QAbs.txt');
writematrix(Edges', 'D:\Data\blue_charis_Edges.txt');
writematrix(QrelLE, 'D:\Data\blue_charis_QrelLE.txt');
writematrix(QrelHE, 'D:\Data\blue_charis_QrelHE.txt');

data.QrelLE = QrelLE;
data.QrelHE = QrelHE;

save(data, 'C:\Users\tolga\Downloads\test_relative\test_relative_AS.mat');

n = size(data.pred_all_q,1);
totalErr = 0;
for i=1:n
    qi = squeeze(data.pred_all_q(i, :,:));
    qgnd = data.labels(i,1:4);
    if(qi(1)<0)
        qi=-qi;
    end
    if(qgnd(1)<0)
        qgnd=-qgnd;
    end
    err1 = t_qdist(qi(1,:),qgnd);
    err2 = t_qdist(qi(2,:),qgnd);
    err3 = t_qdist(qi(3,:),qgnd);
    err = min(min(err1,err2),err3);
    totalErr = totalErr+err;
end
totalErr/n

