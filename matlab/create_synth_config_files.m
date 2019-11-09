
outfolder = 'C:\Users\tolga\Box Sync\gitforks\OptimalTransportSync\py\MMDSync\configs\synth';

bashScpritName1 = [outfolder '/bash1.sh'];
bashScpritName2 = [outfolder '/bash2.sh'];
numGPUs = 3;
numThreads = 4;

% constants
dict = {'model', '''synthetic''';
    'true_prior', '''gaussian''';
    'dtype', '''64''';
    'prior', '''gaussian''';
    'particles_type', '''quaternion''';
    'product_particles', '1';
    'with_couplings', 'False';
    'weights_factor', '0.005';
    'true_bernoulli_noise', '-1.0';
    'eval_loss', '''sinkhorn''';
    'N', '10';
    'log_name', 'synthetic';
    'log_dir', '/nfs/data/michaela/projects/OptSync';
    'batch_size', '10000';
    'num_true_particles', '4';
    'with_weights', '1'};

% will change
completeness = [0.2, 0.5, 0.7];
lr = [0.01 0.05 0.1];
loss = {'mmd', 'sinkhorn'};
optimizer = {'SGD', 'SGD_unconstrained'};
kernel_cost = {'squared_euclidean', 'power_quaternion'};
power = [1.1, 1.5, 2.0];
num_particles = [1,2,4,6,10];
%true_rm_noise_level = [0, 0.05, 0.1, 0.25];
true_rm_noise_level = [0];
allIter=0;

generatedFiles = {};

for i=1:length(completeness)
    cI = completeness(i);
    for j=1:length(loss)
        lossJ = loss{j};
        for k=1:length(optimizer)
            optK = optimizer{k};
            if (strcmp(lossJ, 'sinkhorn') && strcmp(optK, 'SGD_unconstrained'))
                continue;
            end
            for l=1:length(kernel_cost)
                kernelL = kernel_cost{l};
                for m=1:length(power)
                    powerM = power(m);
                    for n=1:length(num_particles)
                        numN = num_particles(n);
                        for o=1:length(true_rm_noise_level)
                            noiseO = true_rm_noise_level(o);
                            for lri=1:length(lr)
                                lrcur = lr(lri);
                                
                                % write dict
                                % writecell(dict,'d:/Data/dict.txt','Delimiter',' : ');
                                fileName = sprintf('%s/config_comp_%g_loss_%s_opt_%s_ker_%s_pow_%g_N_%02d_sigma_%g_lr_%g.yaml', outfolder, cI, lossJ, optK, kernelL, powerM, numN, noiseO, lrcur);
                                fileNameGen = sprintf('config_comp_%g_loss_%s_opt_%s_ker_%s_pow_%g_N_%02d_sigma_%g_lr_%g.yaml', cI, lossJ, optK, kernelL, powerM, numN, noiseO, lrcur);                                
                                fid = fopen(fileName, 'w');
                                for di=1:length(dict)
                                    fprintf(fid, [dict{di,1} ' : ' dict{di,2} '\n']);
                                end
                                fprintf(fid, '\n');
                                fprintf(fid, ['completeness : ' num2str(cI) '\n']);
                                fprintf(fid, ['loss : ' lossJ '\n']);
                                fprintf(fid, ['optimizer : ' optK '\n']);
                                fprintf(fid, ['kernel_cost : ' kernelL '\n']);
                                fprintf(fid, ['power : ' num2str(powerM) '\n']);
                                fprintf(fid, ['num_particles : ' num2str(numN) '\n']);
                                fprintf(fid, ['true_rm_noise_level : ' num2str(noiseO) '\n']);
                                
                                fclose(fid);
                                
                                generatedFiles = [generatedFiles; fileNameGen];
                                
                                % return;
                            end
                        end
                    end
                end
            end
        end
    end    
end

fid = fopen(bashScpritName1, 'w');
for i=1:length(generatedFiles)/2
    fn = generatedFiles{i};
    deviceIndex = mod(i-1,numGPUs);
    threadId = mod(i-1, numThreads);
    if (threadId==(numThreads-1) && deviceIndex == (numGPUs-1))
        bashLine = sprintf('python --device=%d --config_method=configs/%s ;', deviceIndex, fn);
    else
        bashLine = sprintf('python --device=%d --config_method=configs/%s &', deviceIndex, fn);
    end
    fprintf(fid, '%s\n', bashLine);
end
fclose(fid);

fid = fopen(bashScpritName2, 'w');
for j=i+1:length(generatedFiles)
    fn = generatedFiles{j};
    deviceIndex = mod(j-1, numGPUs);
    threadId = mod(j-1, numThreads);
    if (threadId==(numThreads-1) && deviceIndex == (numGPUs-1))
        bashLine = sprintf('python --device=%d --config_method=configs/%s ;', deviceIndex, fn);
    else
        bashLine = sprintf('python --device=%d --config_method=configs/%s &', deviceIndex, fn);
    end
    fprintf(fid, '%s\n', bashLine);
end
fclose(fid);
