import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
import cv2

def sphere(n):
    theta = np.arange(-n, n + 1, 2) / n * np.pi
    phi = np.arange(-n, n + 1, 2).T / n * np.pi / 2

    theta = theta.reshape(1, n + 1)
    phi = phi.reshape(n + 1, 1)

    cosphi = np.cos(phi)

    cosphi[0] = 0
    cosphi[n] = 0

    sintheta = np.sin(theta)
    sintheta[0][0] = 0
    sintheta[0][n] = 0

    x = np.dot(cosphi, np.cos(theta))
    y = np.dot(cosphi, sintheta)
    z = np.dot(np.sin(phi), np.ones((1, n + 1)))

    return x, y, z


def bingham_pdf_3d(x, z1, z2, z3, v1, v2, v3, F):
    Z = F

    cos1 = np.dot(x, v1)
    cos2 = np.dot(x, v2)
    cos3 = np.dot(x, v3)
    p = (1 / Z) * np.exp(z1 * cos1 ** 2 + z2 * cos2 ** 2 + z3 * cos3 ** 2)

    return p


def plot_bingham_3d(V, Z, F, quaternions=None, precision=400):
    [SX, SY, SZ] = sphere(precision)

    n = SX.shape[0]
    C = np.zeros((n, n))
    if V is None:
        fig = plt.figure()
        fig.clear()
        ax = fig.gca(projection='3d')
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zticks([-1, -0.5, 0, 0.5, 1])

        ax.grid(False)
        ax.axis('off')
        surf = ax.plot_surface(SX, SY, SZ, cmap=cm.jet, facecolors=cm.jet(C), linewidth=0, antialiased=True, rstride=1,
                               cstride=1, alpha=0.75)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        surface = buf[:, :, :3][:, :, ::-1]
        return surface

    C = np.zeros((len(V), n, n))
    for i in range(n):
        for j in range(n):
            u = np.array([SX[i, j], SY[i, j], SZ[i, j]])
            for a in np.arange(0, 2 * np.pi, .1):
                q = np.concatenate((np.array([np.cos(a / 2)]), np.sin(a / 2) * u))
                for b in range(len(V)):
                    C[b, i, j] = C[b, i, j] + bingham_pdf_3d(q, Z[b][0], Z[b][1], Z[b][2], V[b][:, 0], V[b][:, 1], V[b][:, 2], F[b])

    C = C / C.max(axis=2).max(axis=1).reshape(len(V), 1, 1)
    C = C.sum(axis=0)


    # for i in range(n):
    #     for j in range(n):
    #         u = np.array([SX[i, j], SY[i, j], SZ[i, j]])
    #         for a in np.arange(0, 2 * np.pi, .1):
    #             q = np.concatenate((np.array([np.cos(a / 2)]), np.sin(a / 2) * u))
    #             C[i, j] = C[i, j] + bingham_pdf_3d(q, Z[0], Z[1], Z[2], V[:, 0], V[:, 1], V[:, 2], F)
    #
    # C = C / C.max()

    fig = plt.figure()
    if quaternions is not None:
        ax = fig.gca(projection='3d')
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_zlim(-1, 1)

        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zticks([-1, -0.5, 0, 0.5, 1])

        ax.grid(False)
        ax.axis('off')

        dist_back = 0

        a = 2 * np.arccos(quaternions[0, 0])
        v_front = quaternions[0, 1:] / np.sin(a / 2)# * 1.1
        a = 2 * np.arccos(quaternions[1, 0])
        v_back = quaternions[1, 1:] / np.sin(a / 2)

        if np.linalg.norm(v_back[:3] - np.asarray([-1,1,-1])) > np.linalg.norm(v_front[:3] - np.asarray([-1,1,-1])):
            tmp = v_back
            v_back = v_front
            v_front = tmp

        ax.scatter([v_back[0]], [v_back[1]], v_back[2], s=200, c='g', alpha=1, marker='x')

        if quaternions.shape[0] > 2:
            a = 2 * np.arccos(quaternions[2, 0])
            v_gt_front = quaternions[2, 1:] / np.sin(a / 2)
            a = 2 * np.arccos(quaternions[3, 0])
            v_gt_back = quaternions[3, 1:] / np.sin(a / 2)

            if np.linalg.norm(v_gt_back[:3] - np.asarray([-1, 1, -1])) > np.linalg.norm(
                            v_gt_front[:3] - np.asarray([-1, 1, -1])):
                tmp = v_gt_back
                v_gt_back = v_gt_front
                v_gt_front = tmp

            ax.scatter([v_gt_back[0]], [v_gt_back[1]], v_gt_back[2], s=200, c='r', alpha=1, marker='x')

        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        back = buf[:,:,:3][:, :, ::-1]

        fig.clear()
        ax = fig.gca(projection='3d')
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zticks([-1, -0.5, 0, 0.5, 1])
        ax.grid(False)
        ax.axis('off')
        dist_front = 0

        ax.scatter([v_front[0]], [v_front[1]], v_front[2], s=200, c='g', alpha=1, marker='x')

        if quaternions.shape[0] > 2:
            ax.scatter([v_gt_front[0]], [v_gt_front[1]], v_gt_front[2], s=200, c='r', alpha=1, marker='x')

        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        front = buf[:, :, :3][:, :, ::-1]

        # if dist_front < dist_back:
        #    tmp = back
        #    back = front
        #    front = tmp

    fig.clear()
    ax = fig.gca(projection='3d')
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])

    ax.grid(False)
    ax.axis('off')
    surf = ax.plot_surface(SX, SY, SZ, cmap=cm.jet, facecolors=cm.jet(C), linewidth=0, antialiased=True, rstride=1,
                           cstride=1, alpha=0.75)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    surface = buf[:, :, :3][:, :, ::-1]

    fig.clear()
    ax = fig.gca(projection='3d')
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])

    ax.grid(False)
    ax.axis('off')

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    layout = buf[:, :, :3][:, :, ::-1]

    if quaternions is not None:

        back_mask = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY) < 255
        neg_surface_mask = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY) >= 255

        surface[neg_surface_mask * back_mask] = back[neg_surface_mask * back_mask]

        surface_mask = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY) < 255
        surface[surface_mask * back_mask] = 0.7 * surface[surface_mask * back_mask] + \
                                                  0.3 * back[surface_mask * back_mask]

        front_mask = cv2.cvtColor(front, cv2.COLOR_BGR2GRAY) < 255
        surface[front_mask] = front[front_mask]

    surface_mask = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY) < 255
    layout[surface_mask] = surface[surface_mask]

    plt.close(fig)

    return layout


def get_bingham(eng, X, GT=None, precision=400):

    Vs, Zs, Fs = [], [], []
    for X_b in X:
        B = eng.bingham_fit(X_b)
        Vs.append(np.asarray(B['V']))
        Zs.append(np.asarray(B['Z'][0]))
        Fs.append(np.asarray(B['F']))

    return plot_bingham_3d(Vs, Zs, Fs, GT, precision)