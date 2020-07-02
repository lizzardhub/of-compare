import numpy as np
import pyDE


class DE:
    estimator = None
    width = None
    height = None
    pixel_precision = None

    def __init__(self, width, height, hier=True, use_fgs=True,
                 fgs_lambda=8000.0, fgs_sigma_color=1.5,
                 me_warming_frame_num=7, use_color_hm=False,
                 min_block_size=2, max_block_size=16):

        assert isinstance(width, int), "Frame width \
                must be integer! width = {}".format(width)
#         assert width % 16 == 0, "Frame width \
#                 must be a multiply 16! width = {}".format(width)
        assert isinstance(height, int), "Frame height \
                must be integer! height = {}".format(height)
#         assert height % 16 == 0, "Frame height \
#                 must be a multiply 16! height = {}".format(height)
        assert isinstance(hier, bool), "Hierarchy parameter \
                must be boolean! hier = {}".format(hier)
        assert isinstance(use_fgs, bool), "Filtration parameter \
                must be boolean! use_fgs = {}".format(use_fgs)
        assert isinstance(me_warming_frame_num, int), "ME Warming parameter \
                must be integer! me_warming_frame_num = {}"\
                .format(me_warming_frame_num)
        assert isinstance(use_color_hm, bool), "Color histogram matching \
                parameter must be boolean! use_color_hm = {}"\
                .format(use_color_hm)
        assert fgs_lambda > 0, "Filtering parameters \
                must be greater than 0! fgs_lambda = ".format(fgs_lambda)
        assert fgs_sigma_color > 0, "Filtering parameters \
                must be greater than 0! fgs_sigma_color = "\
                .format(fgs_sigma_color)

        self.width = width
        self.height = height
        self.estimator = pyDE.DE()
        self.estimator.InitDE(width, height, hier, use_fgs,
                        fgs_lambda, fgs_sigma_color, me_warming_frame_num,
                        use_color_hm, min_block_size, max_block_size)

    def EstimateDisp(self, current_frame, reference_frame):
        if not isinstance(current_frame, np.ndarray):
            current_frame = np.array(current_frame)
        current_frame = current_frame.astype(np.uint8)
        if not isinstance(reference_frame, np.ndarray):
            reference_frame = np.array(reference_frame)
        reference_frame = reference_frame.astype(np.uint8)
        assert current_frame.shape == (
        self.height, self.width, 3), "current_frame.shape must be equal ({}, {}, 3)! current_frame.shape = {}".format(
            self.height, self.width, current_frame.shape)
        assert reference_frame.shape == (self.height, self.width, 3), \
            "reference_frame.shape must be equal ({}, {}, 3)! reference_frame.shape = {}".format(
            self.height, self.width, reference_frame.shape)

        self.estimator.EstimateDisp(current_frame, reference_frame)
#         result = self.estimator.EstimateDisp(current_frame, reference_frame).copy()

#     def Compensate(self, frame):
#         return self.estimator.Compensate(frame).copy()

    def GetDisparityMap(self):
        return self.estimator.GetDisparityMap().copy()

    def GetConfidenceMap(self):
        return self.estimator.GetConfidenceMap().copy()

    def GetRawDisparityMap(self):
        return self.estimator.GetRawDisparityMap().copy()

    def GetRawConfidenceMap(self):
        return self.estimator.GetRawConfidenceMap().copy()

    def GetInputImages(self):
        return self.estimator.GetInputImages().copy()

    def __del__(self):
        if not self.estimator is None:
            self.estimator.DeinitDE()


# def TestDE():
#     import os
#     prefix = os.path.dirname(os.path.realpath(__file__))
#     with open(os.path.join(prefix, "test", "l.png.my"), "rb") as f:
#         data = f.read()
#         l = [i for i in data]
#     l = np.array(l).astype(np.uint8)[:3 * 1024 * 432].reshape((432, 1024, 3))
#
#     with open(os.path.join(prefix, "test", "r.png.my"), "rb") as f:
#         data = f.read()
#         r = [i for i in data]
#     r = np.array(r).astype(np.uint8)[:3 * 1024 * 432].reshape((432, 1024, 3))
#
#     a = pyDE.DE()
#     a.InitDE(1024, 432, 4, 4, 16, "sad", 500, 20)
#     l_to_r_motion_filename = str(os.path.join(prefix, "test", "pyDE_l_to_r.motion"))
#     a.LogDE(l, r, l_to_r_motion_filename)
#     a.DeinitDE()
#     b = pyDE.DE()
#     b.InitDE(1024, 432, 4, 4, 16, "sad", 500, 20)
#     r_to_l_motion_filename = str(os.path.join(prefix, "test", "pyDE_r_to_l.motion"))
#     b.LogDE(r, l, r_to_l_motion_filename)
#     b.DeinitDE()
#
#     with open(os.path.join(prefix, "test", "GT_l_to_r.motion"), "r") as f_gt, open(l_to_r_motion_filename,
#                                                                                    "r") as f_tested:
#         data_gt = f_gt.read().split("\n")
#         data_tested = f_tested.read().split("\n")
#         assert len(data_gt) == len(data_tested), "l_to_r.motion: motion files has different lens"
#         check = [True if gt != tested else False for gt, tested in zip(data_gt, data_tested)]
#         if np.any(check):
#             idx = check.index(True)
#             raise AssertionError("l_to_r.motion: motion files different in line {}".format(idx + 1))
#
#     with open(os.path.join(prefix, "test", "GT_r_to_l.motion"), "r") as f_gt, open(r_to_l_motion_filename,
#                                                                                    "r") as f_tested:
#         data_gt = f_gt.read().split("\n")
#         data_tested = f_tested.read().split("\n")
#         assert len(data_gt) == len(data_tested), "l_to_r.motion: motion files has different lens"
#         check = [True if gt != tested else False for gt, tested in zip(data_gt, data_tested)]
#         if np.any(check):
#             idx = check.index(True)
#             raise AssertionError("l_to_r.motion: motion files different in line {}".format(idx + 1))
