import cv2
import numpy as np
import os


class MacenkoAugmentor:
    @staticmethod
    def get_stain_matrix(I, beta=0.15, alpha=1):
        I = I.reshape((-1, 3))
        I = (I.astype(np.float64) + 1) / 255.0
        OD = -np.log10(I)
        ODhat = OD[np.all(OD > beta, axis=1)]
        if ODhat.shape[0] < 100:
            return None
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        V = eigvecs[:, 1:3]
        Proj = np.dot(ODhat, V)
        phi = np.arctan2(Proj[:, 1], Proj[:, 0])
        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)
        v1 = np.dot(V, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(V, np.array([np.cos(max_phi), np.sin(max_phi)]))
        if v1[0] > v2[0]:
            HE = np.array([v1, v2]).T
        else:
            HE = np.array([v2, v1]).T
        return HE / np.linalg.norm(HE, axis=0)

    def augment(self, image_path, save_path):
        try:
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return False
            img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            HE_matrix = self.get_stain_matrix(img)
            if HE_matrix is None:
                return False

            h, w, c = img.shape
            I_float = (img.reshape((-1, 3)).astype(np.float64) + 1) / 255.0
            OD = -np.log10(I_float)
            C = np.dot(OD, np.linalg.pinv(HE_matrix.T))

            alpha_h = np.random.uniform(0.7, 1.3)
            alpha_e = np.random.uniform(0.7, 1.3)
            C[:, 0] *= alpha_h
            C[:, 1] *= alpha_e

            OD_aug = np.dot(C, HE_matrix.T)
            aug_img = 255.0 * np.exp(-OD_aug * np.log(10))
            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8).reshape((h, w, c))

            cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            return True
        except Exception:
            return False
