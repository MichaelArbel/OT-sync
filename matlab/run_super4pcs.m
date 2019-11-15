
outputSubfolder = 'out';
kbestFolder = 'D:/Data/meas_sync/kbest/ShapeNet2Samples';
commandPre = 'C:/Users/tolga/source/repos/Super4PCS/x64/Release/Super4PCS.exe -i';
%commandPre = 'python run_fgr.py ';
%command = 'C:\Users\tolga\source\repos\Super4PCS\x64\Release\Super4PCS.exe -i input1.ply input2.ply';

folders = dir(kbestFolder);
folders = {folders.name};

fid = fopen('registration2.sh', 'w');

for i=10:length(kbestFolder)
    folder = folders{i};
    subfolder = [kbestFolder '/' folder];
    subfolders = dir(subfolder);
    subfolders = {subfolders.name};
    
    curFolder = [kbestFolder '/' folder '/' subfolders{3} '/point_cloud'];
    %curFolder = [kbestFolder '/' folder];
    pointClouds = dir([curFolder '/*.obj']);
    pointClouds = {pointClouds.name};
    
    N = length(pointClouds);
    for j = 1:N
        shape1 = [curFolder '/' pointClouds{j}];
        for k=1:N
            if (j==k)
                continue;
            end
            shape2 = [curFolder '/' pointClouds{k}];
            outfn = ['out/' pointClouds{j} '_' pointClouds{k} '.pose'];
            command = [commandPre ' ' shape1 ' ' shape2 ' -m ' outfn];
            %system(command);
            fprintf(fid, '%s\n\n', command);
            %outfn = [outputSubfolder '/' pointClouds{j} '_' pointClouds{k} '.ply'];
            %commandRename = ['move output.ply ' outfn];
            %system(commandRename);
        end
    end    
    disp(folder);
    break;
end

fclose(fid);
