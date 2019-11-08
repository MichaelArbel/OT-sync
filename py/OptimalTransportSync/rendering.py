import scipy
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pyrender
import io
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from pyquaternion import Quaternion

plt.close('all')

maxImages = 72

fuze_trimesh = trimesh.load('D:/Data/mug2.ply')
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

#pyrender.Viewer(scene, use_raymond_lighting=True)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=800.0/600.0)
s = np.sqrt(2)/2
light = pyrender.SpotLight(color=np.ones(3)*0.7, intensity=1.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)


scene = pyrender.Scene()
scene.add(mesh)

#plt.subplot(1,2,1)
#plt.axis('off')
#plt.subplot(1,2,2)
#plt.axis('off')

model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')
model.eval()
cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1,512,1,1)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.squeeze()

# build the feature tree
i=0
X=np.zeros((0,512))
print("feature extraction...")
while (i<maxImages):
    fileName = "%s/color_%04d.png" % ("D:/Data/render/views", i)
    f = get_vector(fileName).numpy()
    X = np.vstack((X, f))
    print(i)
    i=i+1

print("read poses...")

Q=np.zeros((0,4))
T=np.zeros((0,3))
i=0
while (i<maxImages):
    fileName = "%s/%04d.pose" % ("D:/Data/render/views", i)
    camera_pose = np.loadtxt(fileName, dtype='f8', delimiter=' ')
    camera_pose[0:3, 1:3] = -1 * camera_pose[0:3, 1:3]
    # project onto so(3) - important for numerical precision when converting to quaternions
    U, s, Vt = np.linalg.svd(camera_pose[0:3,0:3], full_matrices=False)
    V = Vt.T
    R = np.dot(V, U.T)

    if np.linalg.det(R) < 0: # does the current solution use a reflection?
        V[:, -1] *= -1
        s[-1] *= -1
        R = np.dot(V, U.T)
    R=R.T
    camera_pose[0:3, 0:3] = R
    #camera_pose[0:3, 1:3] = -1 * camera_pose[0:3, 1:3]
    q = Quaternion(matrix=camera_pose)
    Q = np.vstack((Q, q.elements))
    T = np.vstack((T, camera_pose[0:3, 3].T))
    i = i + 1

kdtree = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(X)
K = 8

i = 54
ie = i+2
plt.figure()
plt.axis('off')
while (i < ie):
    f = X[i, :]
    qi = Q[i,:]
    ti = np.transpose([T[i, :]])
    Ri = Quaternion(qi[0], qi[1], qi[2], qi[3]).rotation_matrix
    Ti = np.append(np.append(Ri, ti, 1), [[0, 0, 0, 1]], axis=0)

    [distances, indices] = kdtree.kneighbors([f], K+1, return_distance=True)
    print(distances)

    for j in range(0, 9, 1):
        ind = indices[0][j]
        if (distances[0][j] >5):
            continue

        print(ind)
        qj = Q[ind,:]
        tj = np.transpose([T[ind, :]])

        Rj = Quaternion(qj[0], qj[1], qj[2], qj[3]).rotation_matrix
        Tj = np.append(np.append(Rj, tj, 1), [[0,0,0,1]], axis=0)

        Tij = Ti*np.linalg.inv(Tj)

        scene = pyrender.Scene()
        scene.add(mesh)

        node = scene.add(camera, pose=Tj)

        nodeLight = scene.add(light, pose=Tj)

        r = pyrender.OffscreenRenderer(800, 600)
        color, depth = r.render(scene)

        plt.imshow(depth)
        plt.draw()
        plt.show(block=True)

    i=i+1
    continue

    fileName = "%s/%04d.pose" % ("D:/Data/render/views", i)
    camera_pose = np.loadtxt(fileName, dtype='f', delimiter=' ')
    camera_pose[0:3,1:3] = -1*camera_pose[0:3,1:3]
    
    #camera_pose = np.array([
    #    [-0, -0.850651, -0.525731, -1.57719],
    #    [1, -0, 0, 0],
    #    [0, -0.525731, 0.850651, 2.55195],
    #    [0, 0, 0, 1],
    #])
    
    #camera_pose = np.linalg.inv(camera_pose)
    
    node = scene.add(camera, pose=camera_pose)

    nodeLight = scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(800, 600)
    color, depth = r.render(scene)

    plt.imshow(depth)
    plt.draw()
    plt.show(block=True)

    scene.remove_nore(nodeLight)
    scene.remove_node(node)
    
    i=i+1