
partID = [1,4,10,15,25,150,500,999];

for i=1:length(partID)
    fn = sprintf('D:/Data/particles%d.mat',partID(i));
    D=load(fn);
    particles=D.particles;
    w = squeeze(particles.mmdGeodesic.weights);
    particles = squeeze(particles.mmdGeodesic.particles);
    
    data2.q = squeeze(particles(:,5,:));
    data2.w = w';
    data2.z = repmat([-100 -100 -100], length(w), 1);
    plot_bingham_mixture_data(data2,25);
    
    hold on, plot_quaternions(data2.q);
    hold on, plot_quaternions(-data2.q);
    drawnow;
    fn = sprintf('D:/Data/particles%d.png',partID(i));
    %export_fig(fn, '-transparent');
    disp(partID(i))
end


fn = sprintf('D:/Data/particles999.mat', partID(i));
D=load(fn);
particles=D.particles;
w = squeeze(particles.mmdGeodesic.weights);
particles = squeeze(particles.mmdGeodesic.particles);

for i=1:10
        
    data2.q = squeeze(particles(i,:,:));
    data2.w = w';
    data2.z = repmat([-100 -100 -100], length(w), 1);
    plot_bingham_mixture_data(data2,150);
    
    hold on, plot_quaternions(data2.q);
    hold on, plot_quaternions(-data2.q);
    drawnow;
    fn = sprintf('D:/Data/cameras%d.png',partID(i));
    export_fig(fn, '-transparent');
    disp(partID(i))
end