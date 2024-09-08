import numpy as np
import visgeom as vg

from camera import PerspectiveCamera
from measurements import PrecalibratedCameraMeasurementsFixedWorld
from optim import levenberg_marquardt
from visualise_ba import visualise_moba

"""Example 1 - Motion-only Bundle Adjustment"""

np.random.seed(42)

class PrecalibratedMotionOnlyBAObjective:
    """Implements linearisation of motion-only BA objective function"""

    def __init__(self, measurement):
        """Constructs the objective

        :param measurement: A PrecalibratedCameraMeasurementsFixedWorld object.
        """
        self.measurement = measurement

    @staticmethod
    def extract_measurement_jacobian(point_index, pose_state_c_w, measurement):
        """Computes the measurement Jacobian for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param measurement: The measurement
        :return: The measurement Jacobian
        """
        A = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.jac_project_world_to_normalised_wrt_pose_w_c(pose_state_c_w,
                                                                            measurement.x_w[:, [point_index]])

        return A

    @staticmethod
    def extract_measurement_error(point_index, pose_state_c_w, measurement):
        """Computes the measurement error for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param measurement: The measurement
        :return: The measurement error
        """
        b = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.reprojection_error_normalised(pose_state_c_w * measurement.x_w[:, [point_index]],
                                                             measurement.xn[:, [point_index]])

        return b

    def linearise(self, pose_state_w_c):
        """Linearises the objective over all states and measurements

        :param pose_state_w_c: The current camera pose state in the world frame.
        :return:
          A - The full measurement Jacobian
          b - The full measurement error
          cost - The current cost
        """
        num_points = self.measurement.num

        A = np.zeros((2 * num_points, 6))
        b = np.zeros((2 * num_points, 1))

        pose_state_c_w = pose_state_w_c.inverse()

        for j in range(num_points):
            rows = slice(j * 2, (j + 1) * 2)
            A[rows, :] = self.extract_measurement_jacobian(j, pose_state_c_w, self.measurement)
            b[rows, :] = self.extract_measurement_error(j, pose_state_c_w, self.measurement)

        return A, b, b.T.dot(b)


def main():
    # World box.
    points_w = vg.utils.generate_box()

    # Define common camera.
    w = 640
    h = 480
    focal_lengths = 0.75 * h * np.ones((2, 1))
    principal_point = 0.5 * np.array([[w, h]]).T
    camera = PerspectiveCamera(focal_lengths, principal_point)

    # Generate a set of camera measurements.
    true_pose_w_c = PerspectiveCamera.looks_at_pose(np.array([[3, -4, 0]]).T, np.zeros((3, 1)), np.array([[0, 0, 1]]).T)
    measurement = PrecalibratedCameraMeasurementsFixedWorld.generate(camera, true_pose_w_c, points_w)

    # Construct model from measurements.
    model = PrecalibratedMotionOnlyBAObjective(measurement)

    # Perturb camera pose and use as initial state.
    init_pose_wc = true_pose_w_c + 0.3 * np.random.randn(6, 1)

    # Estimate pose in the world frame from point correspondences.
    x, cost, A, b = levenberg_marquardt(init_pose_wc, model)
    cov_x_final = np.linalg.inv(A.T @ A)

    # Print covariance.
    with np.printoptions(precision=3, suppress=True):
        print('Covariance:')
        print(cov_x_final)

    # Visualise
    visualise_moba(true_pose_w_c, points_w, measurement, x, cost)


if __name__ == "__main__":
    main()
