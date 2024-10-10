import numpy as np
from scipy.stats import norm
import torch
import unittest

from src.experiments.synthetic import LinearTemporalSCM, NonLinearTemporalSCM
from src.experiments.real import SemiSyntheticAdultTemporal, SemiSyntheticLoanTemporal, SemiSyntheticCOMPASTemporal, LearnedAdultTemporal, LearnedLoanTemporal, LearnedCOMPASTemporal
from src.scm import SCM
from src.Distributions import Gaussian

class SimpleTemporalSCM(SCM):

    def __init__(self) -> None:
        super().__init__()

        self.U = [
            Gaussian(0, 1),
            Gaussian(0, 1)
        ]

        self.f = [
            lambda X, U1, t: 0.8*X[t-1, :, 0] + U1,
            lambda X, U2, t: 0.8*X[t-1, :, 1] + X[t, :, 0] + U2
        ]

        self.inv_f = [
            lambda X, t: X[t, :, 0] - 0.8*X[t-1, :, 0],
            lambda X, t: X[t, :, 1] - 0.8*X[t-1, :, 1] - X[t, :, 0]
        ]

        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0])+t*0.005,
            lambda X, t: torch.zeros_like(X[t, :, 1])
        ]

        self.inv_trend = [
            lambda X, t: X - (torch.zeros_like(X[0])+t*0.005),
            lambda X, t: X
        ]

        self.mean = torch.zeros(2)
        self.std = torch.ones(2)
    
    def sample_U(self, N, T, S=1):
        U1 = self.U[0].sample((S,T,N))
        U2 = self.U[1].sample((S,T,N))

        return np.stack([U1, U2], axis=-1)

    def label(self, X):
        return np.ones_like(X)
    
    def get_descendant_mask(self, actionable):
        N, D = actionable.shape
        descendant_mask = torch.ones(1, N, D)
        for i in range(D):
            if actionable[0, 0, i] == 1:
                if i == 0:
                    descendant_mask[0, :, [0, 1]] = 0
                elif i == 1:
                    descendant_mask[0, :, [1]] = 0
        return descendant_mask

class TestTemporalCausalModel(unittest.TestCase):

    def setUp(self) -> None:
        self.obj = SimpleTemporalSCM()
        self._class = SimpleTemporalSCM
        self._components = 2
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_shape(self) -> None:
        
        shapes = [
            (10, 10),
            (1, 10),
            (10, 1)
        ]

        for shape in shapes:
             with self.subTest(shape=shape):
                T, N = shape
                result, _, _ = self.obj.generate(
                    N, T
                )
                self.assertEqual(result.shape, (T, N, self._components))

    def test_shape_sample_from(self) -> None:
        
        shapes = [
            11,
            20,
            100
        ]

        result_base, _, _ = self.obj.generate(
                    100, 10
        )

        for shape in shapes:
             with self.subTest(shape=shape):
                T = shape
                result, _, _ = self.obj.generate(
                    100, T, past=torch.Tensor(result_base)
                )
                self.assertEqual(result.shape, (T, 100, self._components))

    def test_generate_with_past(self) -> None:

        result_base, _, _ = self.obj.generate(
                    100, 100
        )

        result_with_past, _, _ = self.obj.generate(
            100, 100,
            past = torch.tensor(result_base[:10, :, :])
        )

        self.assertTrue(
            np.allclose(result_with_past[:10, :, :], result_base[:10, :, :])
        )

        self.assertFalse(
            np.allclose(result_with_past[10, :, :], result_base[10, :, :])
        )

    def test_intervention(self) -> None:
        
        # Since the seed is fixed, the value should
        # match until we reach the same intervention

        result, _, U_original = self.obj.generate(
            1, 10
        )
        self.obj2 = self._class()
        
        # Define the intervention
        intervention = np.zeros((10, 1, len(self.obj2.f)))
        intervention[5, :, 0] = 1
        
        result_interv, _, U_new = self.obj2.generate(
            1, 10, intervention=intervention
        )

        # Check if the exogenous variables have the same value
        self.assertTrue(
            np.allclose(U_original, U_new)
        )

        # Abduction introduces error
        result_a = (torch.tensor(result[5, :, 0])*self.obj2.std)+self.obj2.mean + 1
        result_b = (torch.tensor(result_interv[5, :, 0])*self.obj2.std)+self.obj2.mean
        self.assertTrue(
            np.allclose(result_a, result_b)
        )
    
    def test_counterfactual(self) -> None:

        # Check if we obtain the same distribution
        # if we compute the counterfactual and back

        result, _, U_original = self.obj.generate(
            20, 10
        )

        result_cf, U_abduction = self.obj.counterfactual(
            torch.tensor(result), return_U=True
        )

        with torch.no_grad():
            # Check if the exogenous variables have the same value
            self.assertTrue(
                np.allclose(U_original, U_abduction, atol=1e-5), msg=f"{U_original[0, :, 0, :]} \n {U_abduction[:, 0, :]}"
            )

            self.assertTrue(
                np.allclose(result, result_cf, atol=1e-5)
            )
    
    def test_counterfactual_intervention(self) -> None:

        result, _, _ = self.obj.generate(
            1, 10
        )

        intervention = torch.zeros((10, 1, len(self.obj.f)))
        intervention[5, :, 0] = 1

        result_cf = self.obj.counterfactual(
            torch.tensor(result), intervention
        )

        with torch.no_grad():
            # Abduction introduces approximation errors, so we need
            # to check just if they are "almost" equal
            self.assertTrue(
                np.allclose(
                    result[5, :, 0] + 1, result_cf[5, :, 0].numpy()
                )
            )

class TestKarimiLinear(TestTemporalCausalModel):
     def setUp(self) -> None:
         self.obj = LinearTemporalSCM(0.3)
         self._class = LinearTemporalSCM
         self._components = 3
    
     def tearDown(self) -> None:
         return super().tearDown()

     def test_labels(self) -> None:
         result, _, _ = self.obj.generate(100, 100)
         labels = self.obj.label(result)
         self.assertEqual(labels.shape, (100,100))


class TestKarimiNonLinear(TestTemporalCausalModel):
     def setUp(self) -> None:
         self.obj = NonLinearTemporalSCM()
         self._class = NonLinearTemporalSCM
         self._components = 3
    
     def tearDown(self) -> None:
         return super().tearDown()

class TestKarimiLinearWithNoTrend(TestTemporalCausalModel):
     def setUp(self) -> None:
         self.obj = LinearTemporalSCM(0.0)
         self._class = LinearTemporalSCM
         self._components = 3
    
     def tearDown(self) -> None:
         return super().tearDown()

class TestSemiSyntheticLoan(TestTemporalCausalModel):
     def setUp(self) -> None:
         self.obj = SemiSyntheticLoanTemporal(1.0)
         self._class = SemiSyntheticLoanTemporal
         self._components = 7
    
     def tearDown(self) -> None:
         return super().tearDown()
     
     def test_intervention(self) -> None:
        
        # Since the seed is fixed, the value should
        # match until we reach the same intervention

        result, _, U_original = self.obj.generate(
            1, 10
        )
        self.obj2 = self._class()
        
        # Define the intervention
        intervention = np.zeros((10, 1, len(self.obj2.f)))
        intervention[5, :, 4] = 1
        
        result_interv, _, U_new = self.obj2.generate(
            1, 10, intervention=intervention
        )

        # Check if the exogenous variables have the same value
        self.assertTrue(
            np.allclose(U_original, U_new)
        )

        # Since we standardize I need to perform the test like this
        result_a = (torch.tensor(result[5, :, 4])*self.obj2.std[4])+self.obj2.mean[4] + self.obj2.std[4]
        result_b = (torch.tensor(result_interv[5, :, 4])*self.obj2.std[4])+self.obj2.mean[4]

        self.assertTrue(
            np.allclose(result_a, result_b)
        )

class TestSemisyntheticAdult(TestTemporalCausalModel):
    def setUp(self) -> None:
         self.obj = SemiSyntheticAdultTemporal(1.0, linear=True)
         self.obj.load("data/scms/adult")
         self._class = SemiSyntheticAdultTemporal
         self._components = 6
    
    def tearDown(self) -> None:
         return super().tearDown()
    
    def test_intervention(self) -> None:
        
        # Since the seed is fixed, the value should
        # match until we reach the same intervention

        result, _, U_original = self.obj.generate(
            1, 10
        )
        self.obj2 = self._class()
        self.obj2.load("data/scms/adult")
        
        # Define the intervention
        intervention = torch.zeros((10, 1, len(self.obj2.f)))
        intervention[5, :, 0] = 1
        
        result_interv, _, U_new = self.obj2.generate(
            1, 10, intervention=intervention
        )

        # Check if the exogenous variables have the same value
        self.assertTrue(
            np.allclose(U_original, U_new)
        )

        # Abduction introduces error
        self.assertTrue(
            np.allclose(result[5, :, 0] + 1, result_interv[5, :, 0])
        )

class TestSemisyntheticCOMPAS(TestTemporalCausalModel):
    def setUp(self) -> None:
         self.obj = SemiSyntheticCOMPASTemporal(1.0, linear=True)
         self.obj.load("data/scms/compas")
         self._class = SemiSyntheticCOMPASTemporal
         self._components = 4
    
    def tearDown(self) -> None:
         return super().tearDown()
    
    def test_intervention(self) -> None:
        
        # Since the seed is fixed, the value should
        # match until we reach the same intervention

        result, _, U_original = self.obj.generate(
            1, 10
        )
        self.obj2 = self._class()
        self.obj2.load("data/scms/compas")
        
        # Define the intervention
        intervention = torch.zeros((10, 1, len(self.obj2.f)))
        intervention[5, :, 0] = 1
        
        result_interv, _, U_new = self.obj2.generate(
            1, 10, intervention=intervention
        )

        # Check if the exogenous variables have the same value
        self.assertTrue(
            np.allclose(U_original, U_new)
        )

        # Abduction introduces error
        self.assertTrue(
            np.allclose(result[5, :, 0] + 1, result_interv[5, :, 0])
        )

class TestLearnedAdult(TestTemporalCausalModel):
    def setUp(self) -> None:
         self.obj = LearnedAdultTemporal(linear=True)
         self.obj.load(output_name="adult_test", path="./learned_scms")
         self._class = LearnedAdultTemporal
         self._components = 6
    
    def tearDown(self) -> None:
         return super().tearDown()
    
    def test_intervention(self) -> None:
        
        # Since the seed is fixed, the value should
        # match until we reach the same intervention

        result, _, U_original = self.obj.generate(
            1, 10
        )
        self.obj2 = self._class(linear=True)
        self.obj2.load(output_name="adult_test", path="./learned_scms")
        
        # Define the intervention
        intervention = torch.zeros((10, 1, len(self.obj2.f)))
        intervention[5, :, 0] = 1
        
        result_interv, _, U_new = self.obj2.generate(
            1, 10, intervention=intervention
        )

        # Check if the exogenous variables have the same value
        self.assertTrue(
            np.allclose(U_original, U_new)
        )

        # Abduction introduces error
        self.assertTrue(
            np.allclose(result[5, :, 0] + 1, result_interv[5, :, 0])
        )

class TestLearnedLoan(TestTemporalCausalModel):
    def setUp(self) -> None:
         self.obj = LearnedLoanTemporal(linear=True)
         self.obj.load()
         self._class = LearnedLoanTemporal
         self._components = 7
    
    def tearDown(self) -> None:
         return super().tearDown()
    
    def test_intervention(self) -> None:
        
        # Since the seed is fixed, the value should
        # match until we reach the same intervention

        result, _, U_original = self.obj.generate(
            1, 10
        )
        self.obj2 = self._class(linear=True)
        self.obj2.load()
        
        # Define the intervention
        intervention = torch.zeros((10, 1, len(self.obj2.f)))
        intervention[5, :, 0] = 1
        
        result_interv, _, U_new = self.obj2.generate(
            1, 10, intervention=intervention
        )

        # Check if the exogenous variables have the same value
        self.assertTrue(
            np.allclose(U_original, U_new)
        )

        # Abduction introduces error
        self.assertTrue(
            np.allclose(result[5, :, 0] + 1, result_interv[5, :, 0])
        )

class TestLearnedCOMPAS(TestTemporalCausalModel):
    def setUp(self) -> None:
         self.obj = LearnedCOMPASTemporal(model_type="linear")
         self.obj.load()
         self._class = LearnedCOMPASTemporal
         self._components = 4
    
    def tearDown(self) -> None:
         return super().tearDown()
    
    def test_intervention(self) -> None:
        
        # Since the seed is fixed, the value should
        # match until we reach the same intervention

        result, _, U_original = self.obj.generate(
            1, 10
        )
        self.obj2 = self._class(model_type="linear")
        self.obj2.load()
        
        # Define the intervention
        intervention = torch.zeros((10, 1, len(self.obj2.f)))
        intervention[5, :, 0] = 1
        
        result_interv, _, U_new = self.obj2.generate(
            1, 10, intervention=intervention
        )

        # Check if the exogenous variables have the same value
        self.assertTrue(
            np.allclose(U_original, U_new)
        )

        # Abduction introduces error
        self.assertTrue(
            np.allclose(result[5, :, 0] + 1, result_interv[5, :, 0])
        )

if __name__ == "__main__":
    unittest.main()