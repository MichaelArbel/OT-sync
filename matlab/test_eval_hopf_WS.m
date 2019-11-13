
N = 11;
%
Euler = load('rotation_samples/oim10.eul');

Qequ = zeros(length(Euler),4);
for i=1:length(Euler)
    Qequ(i,:) = euler2q(Euler(i,:));
end

% figure, plot_quaternions(Qequ, zeros(length(Qequ),3), 1,0);

Qequ1q = Qequ(1,:);
Qequ2q = gen_rand_quat(1, 1);
d = sinkhornDistQ(Qequ1q, Qequ2q)

return ;

dg=[];
duni=[];
dsingle =[];
ddouble = [];
dtriple = [];

for i=1:50
    Qgauss = gen_rand_quat(N, 0);
    Quni = gen_rand_quat(N, 1);
    
    % rand sample around single mode
    q0 = gen_rand_quat(1,0);
    Qsingle = sample_around_q(q0, N, 0);
    
    q1 = q0 + 0.15*randn(1,4);
    q1 = q1./norm(q1);
    %q0 = gen_rand_quat(1,0);
    Qdouble = sample_around_q(q0, N/2, 0);
    Qdouble = [Qdouble; sample_around_q(q1, N/2, 0)];
    
    q0 = gen_rand_quat(1,0);
    q1 = q0 + 0.35*randn(1,4);
    q1 = q1./norm(q1);
    q2 = q1 + 0.25*randn(1,4);
    q2 = q2./norm(q2);
    %q0 = gen_rand_quat(1,0);
    N3 =  int32(fix(N/3));
    Qtriple = sample_around_q(q0, N3, 0);
    Qtriple = [Qtriple; sample_around_q(q1, N3, 0)];
    Qtriple = [Qtriple; sample_around_q(q2, N3, 0)];
    
    %Q2 = Qequ(1:20,:)+rand(20,4);
    %Q2 = Q2./repmat(sqrt(dot(Q2,Q2,2)), 1,4);
    
    %d = sinkhornDistQ(Qequ, Qequ)
    d = sinkhornDistQ(Qgauss, Qequ);
    dg = [dg d];
    d = sinkhornDistQ(Quni, Qequ);
    duni = [duni d];
    d = sinkhornDistQ(Qsingle, Qequ);
    dsingle = [dsingle d];
    d = sinkhornDistQ(Qdouble, Qequ);
    ddouble = [ddouble d];
    d = sinkhornDistQ(Qtriple, Qequ);
    dtriple = [dtriple d];
end
figure, plot(dg);
hold on, plot(duni);
hold on, plot(dsingle);
hold on, plot(ddouble);


hold on, plot(dtriple);
legend({'gauss','uni','single','double','triple'});

Qequ2 = Qequ+0.0000001*randn(size(Qequ));
Qequ2 = Qequ2./repmat(sqrt(dot(Qequ2, Qequ2, 2)), 1, 4);
d = sinkhornDistQ(Qequ, Qequ2);
dngd = d;

figure, bar(5*([mean(dg), mean(duni),mean(dsingle), mean(ddouble), mean(dtriple)]-1.9));
figure, bar(([mean(dg), mean(duni),mean(dsingle), mean(ddouble), mean(dtriple)]-mean(dg)));
xticklabels({'gauss','uni','single','double','triple'});