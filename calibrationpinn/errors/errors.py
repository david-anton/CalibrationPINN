from pathlib import Path


class Error(Exception):
    pass


class UnvalidMainConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DirectoryNotFoundError(Error):
    def __init__(self, path_to_directory: Path) -> None:
        self._message = f"The directory {path_to_directory} could not be found"
        super().__init__(self._message)


class FileNotFoundError(Error):
    def __init__(self, path_to_file: Path) -> None:
        self._message = f"The requested file {path_to_file} could not be found!"
        super().__init__(self._message)


class TestConfigurationError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DatasetConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CalibrationDataConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnvalidCalibrationDataError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CalibrationConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnivariateUniformDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MultivariateUniformDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnivariateNormalDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MultivariateNormalDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class IndependentMultivariateNormalDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GammaDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MixedDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class BayesianNNError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class BayesianTrainingError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class FEMConfigurationError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class FEMDomainConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class FEMProblemConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class FEMResultsError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PlottingConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DistanceFunctionConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnvalidGPParametersError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnvalidGPMultivariateNormalError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnvalidGPPriorConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GPMeanNotImplementedError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GPKernelNotImplementedError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GPMeanPriorNotImplementedError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GPKernelPriorNotImplementedError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ParameterSamplingError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class OptimizedModelErrorGPLikelihoodStrategyError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class EMCEEConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)
