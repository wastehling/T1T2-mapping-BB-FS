import numpy as np


#rotation matrix around z axis
def rotate_z(theta):
    '''
    :param theta: rotate by angle in degrees
    :return: rotation matrix around z axis
    '''
    theta_rad = np.deg2rad(theta)
    mat = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                     [np.sin(theta_rad), np.cos(theta_rad), 0],
                     [0, 0, 1]])
    return mat

def rotate_x(alpha):
    '''
    :param alpha: rotate by angle in degrees 
    :return: rotation matrix around x-axis
    '''
    alpha_rad = np.deg2rad(alpha)
    mat = np.array([[1, 0, 0],
                     [0, np.cos(alpha_rad), -np.sin(alpha_rad)],
                     [0, np.sin(alpha_rad), np.cos(alpha_rad)]])
    return mat


def get_rotation_matrix(theta, alpha):
    '''
    :param theta:  angle by which RF pulse is applied
    :param alpha: RF excitation angle
    :return: rotation matrix to excite magnetization
    '''
    #matmul rotates around z axis, then x axis, then z axis
    a = np.matmul(rotate_z(theta), np.matmul(rotate_x(alpha), rotate_z(-theta)))
    # a = rotate_z(theta).dot(rotate_x(alpha).dot(rotate_z(-theta)))
    return a
    # return rotate_z(theta) * rotate_x(alpha) * rotate_z(-theta)


def get_rel_matrix(T1, T2, TR):
    #diagonal matrix
    diag_matrix = np.array([[np.exp(-TR/T2), 0, 0],
                             [0, np.exp(-TR/T2), 0],
                             [0, 0, np.exp(-TR/T1)]])
    return diag_matrix


def test_rot_matrix():
    theta = [0,45,90,180,270]
    alpha = [90,90,90,90,90]

    for idx, t in enumerate(theta):
        print(f'Theta: {t} degrees, Alpha: {alpha[idx]} degrees')
        rot = get_rotation_matrix(t, alpha[idx])
        magn = np.array([0, 0, 1])
        print(f'Magnetization after rotation:')
        print(rot @ magn.T)


if __name__ == '__main__':
    test_rot_matrix()
    print('done')